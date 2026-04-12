"""Layer scheduler for the forward compiler.

Given the current residual stream state and graph metadata, decides what
to compute/cancel in one transformer layer. Returns AttnHeadOp and MLPOp
lists for the weight writer.

Mutates residual_map (allocate, free, reassign) and computed_nodes (add).
"""

from typing import List, Optional, Set, Tuple

from torchwright.compiler.residual_assignment import flatten_concat_nodes
from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.compiler.forward.residual_map import ResidualStreamMap
from torchwright.compiler.forward.weight_writer import AttnHeadOp, MLPOp
from torchwright.graph import Node, Linear, Attn, Add, Concatenate
from torchwright.graph.misc import LiteralValue
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.graph.relu import ReLU


class LayerScheduler:
    """Decides what operations to schedule in one transformer layer.

    Given the current residual-stream allocation and which nodes have
    been computed, picks attention and MLP operations for the next layer.

    Key concepts:
        - **dead node**: all downstream consumers are already computed,
          so its residual-stream columns can be reclaimed.
        - **pressure**: free columns are below 25% of ``d`` — the
          scheduler prioritises operations that free columns over those
          on the critical path.
        - **free add**: an ``Add`` where one input is dead — the dead
          input's columns are reused in-place (no allocation needed).
    """

    def __init__(
        self,
        graph: GraphAnalyzer,
        d: int,
        d_head: int,
        pos_encoding: PosEncoding,
        d_hidden: Optional[int] = None,
    ):
        self.graph = graph
        self.d = d
        self.d_hidden = d if d_hidden is None else d_hidden
        self.d_head = d_head
        self.n_heads = d // d_head
        self.pos_encoding = pos_encoding

    def schedule_layer(
        self, residual_map: ResidualStreamMap, computed_nodes: Set[Node]
    ) -> Tuple[List[AttnHeadOp], List[MLPOp], List[Node]]:
        """Schedule one transformer layer's worth of operations.

        Phases:
            1. Classify ready nodes (free adds, deferred adds, compute-ready).
            2. Attention sublayer: free adds, compute ops (Attn/Linear/Add),
               cancellations of dead nodes.
            3. MLP sublayer: Linear->ReLU->Linear chains, standalone ReLUs,
               constants, bias writes for biased Linears.

        Mutates ``residual_map`` (allocate/free/reassign) and
        ``computed_nodes`` (add newly computed nodes).

        Returns:
            ``(attn_ops, mlp_ops, biased_linears)`` lists for the weight writer.
        """
        # --- 1. Classify ready nodes ---
        all_ready = self.graph.get_ready_nodes(computed_nodes)

        ready = set()
        free_adds = []
        deferred_adds = []
        for node in all_ready:
            if isinstance(node, Add):
                a0, a1 = node.inputs
                d0 = self._is_dead_for_add(a0, node, computed_nodes)
                d1 = self._is_dead_for_add(a1, node, computed_nodes)
                if d0 or d1:
                    free_adds.append(node)
                else:
                    deferred_adds.append(node)
            elif isinstance(node, (Attn, Linear, ReLU, LiteralValue)):
                ready.add(node)
            # else: skip unschedulable source nodes (InputNode, Embedding, etc.)

        dead = self._find_dead_nodes(residual_map, computed_nodes)
        chains = self._detect_chains(ready)

        # Remove chain-internal nodes from ready so they're not double-scheduled.
        # L1 with fanout stays in ready for standalone attention scheduling.
        for l1, relu, l2, d_hidden, exclusive in chains:
            ready.discard(l2)
            ready.discard(relu)
            if exclusive:
                ready.discard(l1)

        had_schedulable = (
            bool(ready) or bool(free_adds) or bool(deferred_adds) or bool(chains)
        )

        # --- 2. Attention sublayer ---
        attn_ops, biased_linears = self._schedule_attn_sublayer(
            ready, dead, free_adds, deferred_adds, residual_map, computed_nodes
        )

        # --- 2.5. Re-check readiness after attention ---
        # Nodes computed by attention may unlock MLP-eligible nodes in the same
        # layer (the MLP sublayer reads x + attn(x), so it sees attention results).
        newly_ready = self.graph.get_ready_nodes(computed_nodes) - all_ready
        for node in newly_ready:
            if isinstance(node, Add):
                a0, a1 = node.inputs
                d0 = self._is_dead_for_add(a0, node, computed_nodes)
                d1 = self._is_dead_for_add(a1, node, computed_nodes)
                if not (d0 or d1):
                    continue  # deferred add, skip
            if isinstance(node, (Linear, ReLU, LiteralValue)):
                ready.add(node)

        new_chains = self._detect_chains(ready)
        for l1, relu, l2, d_hidden, exclusive in new_chains:
            ready.discard(l2)
            ready.discard(relu)
            if exclusive:
                ready.discard(l1)
        chains.extend(new_chains)

        # --- 3. MLP sublayer ---
        mlp_ops = self._schedule_mlp_sublayer(
            ready, chains, biased_linears, residual_map, computed_nodes
        )

        # --- 4. Progress check ---
        # Only raise if there were ready/schedulable nodes but nothing got scheduled
        # (actual deadlock). If nothing was ready, the caller just needs to provide
        # more inputs or call again after state changes.
        if not attn_ops and not mlp_ops and had_schedulable:
            remaining = self.graph.get_all_nodes() - computed_nodes
            remaining = {n for n in remaining if not isinstance(n, Concatenate)}
            if remaining:
                raise RuntimeError(
                    f"No progress: {len(remaining)} nodes remaining, "
                    f"{residual_map.get_free_count()} free columns"
                )

        return attn_ops, mlp_ops, biased_linears

    # ------------------------------------------------------------------
    # Attention sublayer
    # ------------------------------------------------------------------

    def _schedule_attn_sublayer(
        self, ready, dead, free_adds, deferred_adds, residual_map, computed_nodes
    ):
        attn_ops = []
        biased_linears = []
        heads_used = 0

        # 2a. Free Adds (highest priority — no allocation needed)
        # Snapshot computed_nodes so dead-for-add checks are consistent across
        # the entire batch. Without this, earlier add_into ops add their Add to
        # computed_nodes, which can flip a shared node from "live" to "dead" on
        # a later iteration — reassigning its columns and orphaning earlier ops.
        computed_snapshot = set(computed_nodes)
        add_into_live_addends = set()
        for add_node in sorted(free_adds, key=self._critical_path_key):
            if heads_used >= self.n_heads:
                break
            a0, a1 = add_node.inputs
            d0 = self._is_dead_for_add(a0, add_node, computed_snapshot)
            d1 = self._is_dead_for_add(a1, add_node, computed_snapshot)
            dead_addend = a0 if d0 else a1
            live_addend = a1 if d0 else a0
            n_heads = (len(live_addend) + self.d_head - 1) // self.d_head
            if heads_used + n_heads > self.n_heads:
                continue
            target_cols = residual_map.get_indices(dead_addend)
            attn_ops.append(AttnHeadOp("add_into", add_node, target_cols))
            residual_map.reassign(dead_addend, add_node)
            computed_nodes.add(add_node)
            add_into_live_addends.add(live_addend)
            heads_used += n_heads

        # Build compute candidates: Attn nodes, standalone Linears, deferred Adds
        compute_candidates = []
        for node in ready:
            if isinstance(node, Attn):
                n_heads = (node.d_v + self.d_head - 1) // self.d_head
                compute_candidates.append(("compute_attn", node, n_heads))
            elif isinstance(node, Linear) and not isinstance(node.inputs[0], ReLU):
                n_heads = self._heads_for_linear(node)
                compute_candidates.append(("compute_linear", node, n_heads))
        # Deferred Adds: neither input is dead, so we can't use add_into.
        # Instead, copy both inputs to fresh columns via two groups of heads.
        for node in deferred_adds:
            n_heads = 2 * self._heads_for_node(node)
            compute_candidates.append(("compute_add", node, n_heads))

        # Sort: Attn first; under column pressure prefer nodes that free columns,
        # otherwise maximize parallelism via critical path.
        under_pressure = residual_map.get_free_count() < self.d // 4
        if under_pressure:
            compute_candidates.sort(
                key=lambda t: (
                    0 if t[0] == "compute_attn" else 1,
                    self._net_column_cost(t[1], computed_nodes, residual_map),
                    -self.graph.get_critical_path_length(t[1]),
                )
            )
        else:
            compute_candidates.sort(
                key=lambda t: (
                    0 if t[0] == "compute_attn" else 1,
                    -self.graph.get_critical_path_length(t[1]),
                )
            )

        # Cancellation candidates (exclude live addends of add_into ops)
        cancel_candidates = [
            n
            for n in dead
            if n is not self.pos_encoding and n not in add_into_live_addends
        ]
        cancel_candidates.sort(key=lambda n: -len(n))  # largest first

        # 2b-2d. Schedule compute ops with cancellation promotion
        for op_type, node, n_heads_needed in compute_candidates:
            if heads_used + n_heads_needed > self.n_heads:
                continue
            target_cols = self._try_allocate(node, residual_map)

            # Promotion: cancel dead nodes to free space
            while (
                target_cols is None
                and cancel_candidates
                and heads_used + n_heads_needed < self.n_heads
            ):
                cn = cancel_candidates.pop(0)
                cn_heads = (len(cn) + self.d_head - 1) // self.d_head
                if heads_used + n_heads_needed + cn_heads > self.n_heads:
                    continue
                attn_ops.append(AttnHeadOp("cancel", cn, residual_map.get_indices(cn)))
                residual_map.free(cn)
                heads_used += cn_heads
                target_cols = self._try_allocate(node, residual_map)

            if target_cols is None:
                continue

            attn_ops.append(AttnHeadOp(op_type, node, target_cols))
            heads_used += n_heads_needed
            computed_nodes.add(node)
            ready.discard(node)

            if (
                op_type == "compute_linear"
                and isinstance(node, Linear)
                and not self._has_zero_bias(node)
            ):
                biased_linears.append(node)

        # 2e. Remaining cancellations
        for cn in cancel_candidates:
            cn_heads = (len(cn) + self.d_head - 1) // self.d_head
            if heads_used + cn_heads > self.n_heads:
                continue
            attn_ops.append(AttnHeadOp("cancel", cn, residual_map.get_indices(cn)))
            residual_map.free(cn)
            heads_used += cn_heads

        return attn_ops, biased_linears

    # ------------------------------------------------------------------
    # MLP sublayer
    # ------------------------------------------------------------------

    def _schedule_mlp_sublayer(
        self, ready, chains, biased_linears, residual_map, computed_nodes
    ):
        mlp_ops = []
        next_slot = 0

        # 3a. L->R->L chains
        under_pressure = residual_map.get_free_count() < self.d // 4
        if under_pressure:
            chains.sort(
                key=lambda c: (
                    self._chain_net_column_cost(c, computed_nodes, residual_map),
                    -self.graph.get_critical_path_length(c[2]),
                )
            )
        else:
            chains.sort(key=lambda c: -self.graph.get_critical_path_length(c[2]))
        # NOTE: ``d_hidden`` here is the per-chain hidden width (``len(relu)``).
        # ``self.d_hidden`` is the layer-wide MLP hidden pool size.  Same name,
        # different scopes — chains are packed into the layer pool.
        for l1, relu, l2, d_hidden, exclusive in chains:
            if next_slot + d_hidden > self.d_hidden:
                continue
            target_cols = self._try_allocate(l2, residual_map)
            if target_cols is None:
                continue
            mlp_slots = list(range(next_slot, next_slot + d_hidden))
            next_slot += d_hidden
            mlp_ops.append(MLPOp("compute_relu", l2, target_cols, mlp_slots))
            computed_nodes.update({l1, relu, l2})

            # L1 with fanout: also allocate L1 in residual stream
            if not exclusive and not residual_map.is_allocated(l1):
                self._try_allocate(l1, residual_map)

        # 3b. Standalone ReLU (not part of chain)
        standalone_relus = sorted(
            [n for n in ready if isinstance(n, ReLU)],
            key=(
                (
                    lambda n: (
                        self._net_column_cost(n, computed_nodes, residual_map),
                        self._critical_path_key(n),
                    )
                )
                if under_pressure
                else self._critical_path_key
            ),
        )
        for node in standalone_relus:
            d_relu = len(node)
            if next_slot + d_relu > self.d_hidden:
                continue
            target_cols = self._try_allocate(node, residual_map)
            if target_cols is None:
                continue
            mlp_slots = list(range(next_slot, next_slot + d_relu))
            next_slot += d_relu
            mlp_ops.append(
                MLPOp("compute_standalone_relu", node, target_cols, mlp_slots)
            )
            computed_nodes.add(node)

        # 3c. LiteralValues (no slot cost)
        constants = sorted(
            [n for n in ready if isinstance(n, LiteralValue)],
            key=self._critical_path_key,
        )
        for node in constants:
            target_cols = self._try_allocate(node, residual_map)
            if target_cols is None:
                continue
            mlp_ops.append(MLPOp("compute_literal_value", node, target_cols, []))
            computed_nodes.add(node)

        # 3d. Bias writes for biased Linears scheduled in attention sublayer
        for node in biased_linears:
            target_cols = residual_map.get_indices(node)
            mlp_ops.append(MLPOp("compute_bias", node, target_cols, []))

        return mlp_ops

    # ------------------------------------------------------------------
    # Chain detection
    # ------------------------------------------------------------------

    def _detect_chains(self, ready):
        """Detect L1->ReLU->L2 chains by walking forward from ready Linears.

        Returns list of (l1, relu, l2, d_hidden, exclusive).
        exclusive means L1 has no effective consumers other than the ReLU.
        """
        chains = []
        seen_relus = set()

        for node in ready:
            if not isinstance(node, Linear):
                continue
            l1 = node

            for consumer in self.graph.get_consumers(l1):
                if not isinstance(consumer, ReLU) or consumer in seen_relus:
                    continue
                relu = consumer

                # ReLU must have exactly one effective consumer that's a Linear
                relu_eff = self._get_effective_consumers(relu)
                l2_candidates = [c for c in relu_eff if isinstance(c, Linear)]
                if len(relu_eff) != 1 or len(l2_candidates) != 1:
                    continue
                l2 = l2_candidates[0]
                if l2.inputs[0] is not relu:
                    continue

                l1_eff = self._get_effective_consumers(l1)
                exclusive = l1_eff == {relu}
                chains.append((l1, relu, l2, len(relu), exclusive))
                seen_relus.add(relu)
                break  # one chain per L1

        return chains

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _net_column_cost(
        self, node: Node, computed_nodes: Set[Node], residual_map: ResidualStreamMap
    ) -> int:
        """Net residual stream columns consumed by scheduling this node.

        Positive = net consumer, negative = net freer (frees more than it allocates).
        """
        cost = len(node)
        for inp in node.inputs:
            leaves = (
                flatten_concat_nodes([inp]) if isinstance(inp, Concatenate) else [inp]
            )
            for leaf in leaves:
                if leaf is self.pos_encoding:
                    continue
                if not residual_map.is_allocated(leaf):
                    continue
                remaining = (
                    self._get_effective_consumers(leaf) - computed_nodes - {node}
                )
                if not remaining:
                    cost -= len(leaf)
        return cost

    def _chain_net_column_cost(
        self,
        chain: tuple,
        computed_nodes: Set[Node],
        residual_map: ResidualStreamMap,
    ) -> int:
        """Net column cost of an L1->ReLU->L2 chain.

        Only L2 is allocated in the residual stream (L1/ReLU use MLP slots).
        Non-exclusive L1 is also allocated separately.
        """
        l1, relu, l2, d_hidden, exclusive = chain
        cost = len(l2)
        if not exclusive and not residual_map.is_allocated(l1):
            cost += len(l1)

        hypothetical = computed_nodes | {l1, relu, l2}
        for inp in l1.inputs:
            leaves = (
                flatten_concat_nodes([inp]) if isinstance(inp, Concatenate) else [inp]
            )
            for leaf in leaves:
                if leaf is self.pos_encoding:
                    continue
                if not residual_map.is_allocated(leaf):
                    continue
                remaining = self._get_effective_consumers(leaf) - hypothetical
                if not remaining:
                    cost -= len(leaf)
        return cost

    def _get_effective_consumers(self, node: Node) -> Set[Node]:
        """Get consumers, resolving through Concatenate nodes.

        Terminal Concatenates (no consumers, i.e. output nodes) are kept
        as effective consumers so their children aren't freed prematurely.
        """
        result = set()
        for consumer in self.graph.get_consumers(node):
            if isinstance(consumer, Concatenate):
                resolved = self._get_effective_consumers(consumer)
                if resolved:
                    result |= resolved
                else:
                    # Terminal Concatenate (output node) — its children's
                    # columns must stay allocated until compilation ends.
                    result.add(consumer)
            else:
                result.add(consumer)
        return result

    def _is_dead(self, node: Node, computed_nodes: Set[Node]) -> bool:
        if node is self.pos_encoding:
            return False
        if node not in self.graph.get_all_nodes():
            return False
        return self._get_effective_consumers(node).issubset(computed_nodes)

    def _is_dead_for_add(
        self, addend: Node, add_node: Add, computed_nodes: Set[Node]
    ) -> bool:
        """True if all effective consumers of addend, except add_node, are computed.

        Concatenate nodes can't be dead addends — they aren't allocated in the
        residual stream, so their columns can't be reused for add_into.
        """
        if addend is self.pos_encoding:
            return False
        if isinstance(addend, Concatenate):
            return False
        effective = self._get_effective_consumers(addend)
        return (effective - {add_node}).issubset(computed_nodes)

    def _find_dead_nodes(
        self, residual_map: ResidualStreamMap, computed_nodes: Set[Node]
    ) -> List[Node]:
        graph_nodes = self.graph.get_all_nodes()
        dead = []
        for node in residual_map.get_allocated_nodes():
            if node not in graph_nodes:
                continue
            if node not in computed_nodes:
                continue
            if self._is_dead(node, computed_nodes):
                dead.append(node)
        return dead

    def _heads_for_node(self, node: Node) -> int:
        """Number of attention heads needed to copy a node's output."""
        return (len(node) + self.d_head - 1) // self.d_head

    def _heads_for_linear(self, node: Linear) -> int:
        """Number of attention heads needed for a standalone Linear."""
        d_input = len(node.inputs[0])
        return (d_input + self.d_head - 1) // self.d_head

    def _has_zero_bias(self, node: Linear) -> bool:
        return node.output_bias.abs().sum().item() == 0

    def _try_allocate(
        self, node: Node, residual_map: ResidualStreamMap
    ) -> Optional[List[int]]:
        if len(node) > residual_map.get_free_count():
            return None
        return residual_map.allocate(node)

    def _critical_path_key(self, node: Node):
        return -self.graph.get_critical_path_length(node)
