"""Layer scheduler for the forward compiler.

Given the current residual stream state and graph metadata, decides what
to compute/cancel in one transformer layer. Returns AttnHeadOp and FFNOp
lists for the weight writer.

Mutates residual_map (allocate, free, reassign) and computed_nodes (add).
"""

from typing import List, Optional, Set, Tuple

from modelscriptor.compiler.forward.graph_analysis import GraphAnalyzer
from modelscriptor.compiler.forward.residual_map import ResidualStreamMap
from modelscriptor.compiler.forward.weight_writer import AttnHeadOp, FFNOp
from modelscriptor.graph import Node, Linear, Attn, Add, Concatenate
from modelscriptor.graph.misc import Constant
from modelscriptor.graph.pos_encoding import PosEncoding
from modelscriptor.graph.relu import ReLU


class LayerScheduler:
    def __init__(self, graph: GraphAnalyzer, d: int, d_head: int,
                 pos_encoding: PosEncoding):
        self.graph = graph
        self.d = d
        self.d_head = d_head
        self.n_heads = d // d_head
        self.pos_encoding = pos_encoding

    def schedule_layer(
        self, residual_map: ResidualStreamMap, computed_nodes: Set[Node]
    ) -> Tuple[List[AttnHeadOp], List[FFNOp]]:
        # --- 1. Classify ready nodes ---
        all_ready = self.graph.get_ready_nodes(computed_nodes)

        ready = set()
        free_adds = []
        for node in all_ready:
            if isinstance(node, Add):
                a0, a1 = node.inputs
                d0 = self._is_dead_for_add(a0, node, computed_nodes)
                d1 = self._is_dead_for_add(a1, node, computed_nodes)
                if d0 or d1:
                    free_adds.append(node)
                # else: deferred, skip
            elif isinstance(node, (Attn, Linear, ReLU, Constant)):
                ready.add(node)
            # else: skip unschedulable source nodes (InputNode, Embedding, etc.)

        dead = self._find_dead_nodes(residual_map, computed_nodes)
        chains = self._detect_chains(ready)

        # Remove chain-internal nodes from ready so they're not double-scheduled.
        # L1 with fanout stays in ready for standalone attention scheduling.
        for l1, relu, l2, d_int, exclusive in chains:
            ready.discard(l2)
            ready.discard(relu)
            if exclusive:
                ready.discard(l1)

        had_schedulable = bool(ready) or bool(free_adds) or bool(chains)

        # --- 2. Attention sublayer ---
        attn_ops, biased_linears = self._schedule_attn_sublayer(
            ready, dead, free_adds, residual_map, computed_nodes
        )

        # --- 3. FFN sublayer ---
        ffn_ops = self._schedule_ffn_sublayer(
            ready, chains, biased_linears, residual_map, computed_nodes
        )

        # --- 4. Progress check ---
        # Only raise if there were ready/schedulable nodes but nothing got scheduled
        # (actual deadlock). If nothing was ready, the caller just needs to provide
        # more inputs or call again after state changes.
        if not attn_ops and not ffn_ops and had_schedulable:
            remaining = self.graph.get_all_nodes() - computed_nodes
            remaining = {n for n in remaining if not isinstance(n, Concatenate)}
            if remaining:
                raise RuntimeError(
                    f"No progress: {len(remaining)} nodes remaining, "
                    f"{residual_map.get_free_count()} free columns"
                )

        return attn_ops, ffn_ops

    # ------------------------------------------------------------------
    # Attention sublayer
    # ------------------------------------------------------------------

    def _schedule_attn_sublayer(self, ready, dead, free_adds, residual_map,
                                computed_nodes):
        attn_ops = []
        biased_linears = []
        heads_used = 0

        # 2a. Free Adds (highest priority — no allocation needed)
        for add_node in sorted(free_adds, key=self._critical_path_key):
            if heads_used >= self.n_heads:
                break
            a0, a1 = add_node.inputs
            d0 = self._is_dead_for_add(a0, add_node, computed_nodes)
            d1 = self._is_dead_for_add(a1, add_node, computed_nodes)
            dead_addend = a0 if d0 else a1
            live_addend = a1 if d0 else a0
            if len(live_addend) > self.d_head:
                continue
            target_cols = residual_map.get_indices(dead_addend)
            attn_ops.append(AttnHeadOp("add_into", add_node, target_cols))
            residual_map.reassign(dead_addend, add_node)
            computed_nodes.add(add_node)
            heads_used += 1

        # Build compute candidates: Attn nodes, then standalone Linears
        compute_candidates = []
        for node in ready:
            if isinstance(node, Attn):
                compute_candidates.append(("compute_attn", node))
            elif (isinstance(node, Linear)
                  and not isinstance(node.inputs[0], ReLU)
                  and len(node.inputs[0]) <= self.d_head):
                compute_candidates.append(("compute_linear", node))

        # Sort: Attn first, then by critical path
        compute_candidates.sort(key=lambda pair: (
            0 if pair[0] == "compute_attn" else 1,
            -self.graph.get_critical_path_length(pair[1]),
        ))

        # Cancellation candidates
        cancel_candidates = [
            n for n in dead
            if len(n) <= self.d_head and n is not self.pos_encoding
        ]
        cancel_candidates.sort(key=lambda n: -len(n))  # largest first

        # 2b-2d. Schedule compute ops with cancellation promotion
        for op_type, node in compute_candidates:
            if heads_used >= self.n_heads:
                break
            target_cols = self._try_allocate(node, residual_map)

            # Promotion: cancel dead nodes to free space
            while (target_cols is None and cancel_candidates
                   and heads_used < self.n_heads - 1):
                cn = cancel_candidates.pop(0)
                attn_ops.append(
                    AttnHeadOp("cancel", cn, residual_map.get_indices(cn)))
                residual_map.free(cn)
                heads_used += 1
                target_cols = self._try_allocate(node, residual_map)

            if target_cols is None:
                continue

            attn_ops.append(AttnHeadOp(op_type, node, target_cols))
            heads_used += 1
            computed_nodes.add(node)
            ready.discard(node)

            if (op_type == "compute_linear" and isinstance(node, Linear)
                    and not self._has_zero_bias(node)):
                biased_linears.append(node)

        # 2e. Remaining cancellations
        for cn in cancel_candidates:
            if heads_used >= self.n_heads:
                break
            attn_ops.append(
                AttnHeadOp("cancel", cn, residual_map.get_indices(cn)))
            residual_map.free(cn)
            heads_used += 1

        return attn_ops, biased_linears

    # ------------------------------------------------------------------
    # FFN sublayer
    # ------------------------------------------------------------------

    def _schedule_ffn_sublayer(self, ready, chains, biased_linears,
                               residual_map, computed_nodes):
        ffn_ops = []
        next_slot = 0

        # 3a. L->R->L chains
        chains.sort(key=lambda c: -self.graph.get_critical_path_length(c[2]))
        for l1, relu, l2, d_int, exclusive in chains:
            if next_slot + d_int > self.d:
                continue
            target_cols = self._try_allocate(l2, residual_map)
            if target_cols is None:
                continue
            ffn_slots = list(range(next_slot, next_slot + d_int))
            next_slot += d_int
            ffn_ops.append(FFNOp("compute_relu", l2, target_cols, ffn_slots))
            computed_nodes.update({l1, relu, l2})

            # L1 with fanout: also allocate L1 in residual stream
            if not exclusive and not residual_map.is_allocated(l1):
                self._try_allocate(l1, residual_map)

        # 3b. Standalone ReLU (not part of chain)
        standalone_relus = sorted(
            [n for n in ready if isinstance(n, ReLU)],
            key=self._critical_path_key,
        )
        for node in standalone_relus:
            d_relu = len(node)
            if next_slot + d_relu > self.d:
                continue
            target_cols = self._try_allocate(node, residual_map)
            if target_cols is None:
                continue
            ffn_slots = list(range(next_slot, next_slot + d_relu))
            next_slot += d_relu
            ffn_ops.append(FFNOp("compute_standalone_relu", node,
                                 target_cols, ffn_slots))
            computed_nodes.add(node)

        # 3c. Constants (no slot cost)
        constants = sorted(
            [n for n in ready if isinstance(n, Constant)],
            key=self._critical_path_key,
        )
        for node in constants:
            target_cols = self._try_allocate(node, residual_map)
            if target_cols is None:
                continue
            ffn_ops.append(FFNOp("compute_constant", node, target_cols, []))
            computed_nodes.add(node)

        # 3d. Bias writes for biased Linears scheduled in attention sublayer
        for node in biased_linears:
            target_cols = residual_map.get_indices(node)
            ffn_ops.append(FFNOp("compute_bias", node, target_cols, []))

        return ffn_ops

    # ------------------------------------------------------------------
    # Chain detection
    # ------------------------------------------------------------------

    def _detect_chains(self, ready):
        """Detect L1->ReLU->L2 chains by walking forward from ready Linears.

        Returns list of (l1, relu, l2, d_intermediate, exclusive).
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

    def _get_effective_consumers(self, node: Node) -> Set[Node]:
        """Get consumers, resolving through Concatenate nodes."""
        result = set()
        for consumer in self.graph.get_consumers(node):
            if isinstance(consumer, Concatenate):
                result |= self._get_effective_consumers(consumer)
            else:
                result.add(consumer)
        return result

    def _is_dead(self, node: Node, computed_nodes: Set[Node]) -> bool:
        if node is self.pos_encoding:
            return False
        if node not in self.graph.get_all_nodes():
            return False
        return self._get_effective_consumers(node).issubset(computed_nodes)

    def _is_dead_for_add(self, addend: Node, add_node: Add,
                         computed_nodes: Set[Node]) -> bool:
        """True if all effective consumers of addend, except add_node, are computed."""
        if addend is self.pos_encoding:
            return False
        effective = self._get_effective_consumers(addend)
        return (effective - {add_node}).issubset(computed_nodes)

    def _find_dead_nodes(self, residual_map: ResidualStreamMap,
                         computed_nodes: Set[Node]) -> List[Node]:
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

    def _has_zero_bias(self, node: Linear) -> bool:
        return node.output_bias.abs().sum().item() == 0

    def _try_allocate(self, node: Node,
                      residual_map: ResidualStreamMap) -> Optional[List[int]]:
        if len(node) > residual_map.get_free_count():
            return None
        return residual_map.allocate(node)

    def _critical_path_key(self, node: Node):
        return -self.graph.get_critical_path_length(node)
