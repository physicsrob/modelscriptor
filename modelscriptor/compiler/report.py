import os
from datetime import datetime
from typing import List, Optional, Tuple, Union

from modelscriptor.compiler.components.component import Component
from modelscriptor.compiler.components.skip import SkipLayerComponent
from modelscriptor.compiler.groups.attn_sublayer import AttnSubLayer
from modelscriptor.compiler.groups.ffn_sublayer import FFNSubLayer
from modelscriptor.compiler.groups.transformer_layer import TransformerLayer
from modelscriptor.compiler.res_state import ResState
from jinja2 import Environment, FileSystemLoader
from jinja2 import Environment, FileSystemLoader

from modelscriptor.compiler.transformer import HeadlessTransformer
from modelscriptor.graph import Node

current_module_directory = os.path.dirname(os.path.abspath(__file__))
template_directory = os.path.join(current_module_directory, "report_templates")


def render_res_state(residual_state: ResState, name: str):
    # Prepare node span data. Span is defined as (node, min_index, max_index).
    sorted_node_spans = [
        (
            node,
            min(residual_state.get_node_indices(node)),
            max(residual_state.get_node_indices(node)),
        )
        for node in residual_state.get_nodes()
    ]

    # Sort spans by their starting index
    sorted_node_spans.sort(key=lambda span: span[1])

    # Initialize the list that will hold both node and empty spans
    complete_spans: List[Tuple[Optional[str], Optional[int], int]] = []
    last_span_end = -1  # Last ending index of the previous span

    # Fill in node and empty spans
    for node, start_idx, end_idx in sorted_node_spans:
        if start_idx - last_span_end > 1:
            complete_spans.append((None, None, start_idx - last_span_end - 1))
        complete_spans.append((repr(node), id(node), end_idx - start_idx + 1))
        last_span_end = end_idx

    # Add a final empty span if needed
    if last_span_end < residual_state.d:
        complete_spans.append((None, None, residual_state.d - last_span_end))

    env = Environment(loader=FileSystemLoader(template_directory))
    template = env.get_template("res_state_template.html")

    # Populate and render the template
    return template.render(
        indices=list(range(residual_state.d)), node_spans=complete_spans, name=name
    )


def render_node(node: Node):
    env = Environment(loader=FileSystemLoader(template_directory))
    template = env.get_template("node_template.html")
    input_nodes = [(id(n), repr(n)) for n in node.inputs]
    # Populate and render the template
    return template.render(
        node_name=repr(node), node_id=id(node), input_nodes=input_nodes
    )


def render_component(component: Component):
    env = Environment(loader=FileSystemLoader(template_directory))
    template = env.get_template("component_template.html")

    # Populate and render the template
    return template.render(
        component_name=repr(component),
        in_state=render_res_state(component.in_state, name="Input"),
        skip_state=render_res_state(component.skip_state, name="Skip")
        if isinstance(component, SkipLayerComponent)
        else None,
        out_state=render_res_state(component.out_state, name="Output"),
        nodes=[render_node(n) for n in component.out_state.get_nodes()],
    )


def render_ffn_layer(layer: FFNSubLayer, layer_idx: int):
    env = Environment(loader=FileSystemLoader(template_directory))

    template = env.get_template("layer_template.html")

    # Populate and render the template
    return template.render(
        layer_index=layer_idx,
        layer_type="FFN",
        components=[
            render_component(c)
            for c in [layer.linear1, layer.relu, layer.linear2, layer.skip]
        ],
        in_state=render_res_state(layer.in_state, "Layer Input"),
        out_state=render_res_state(layer.out_state, "Layer Output"),
    )


def render_attn_layer(layer: AttnSubLayer, layer_idx: int):
    env = Environment(loader=FileSystemLoader(template_directory))

    template = env.get_template("layer_template.html")

    # Populate and render the template
    return template.render(
        layer_index=layer_idx,
        layer_type="Attention",
        components=[render_component(c) for c in [layer.attn, layer.skip]],
        in_state=render_res_state(layer.in_state, "Layer Input"),
        out_state=render_res_state(layer.out_state, "Layer Output"),
    )


def graph_stats(nodes: List[Node]) -> Tuple[int, int]:
    param_cnt = 0
    node_cnt = 0

    for node in nodes:
        param_cnt += node.num_params()
        node_cnt += 1
        input_param_cnt, input_node_cnt = graph_stats(node.inputs)
        param_cnt += input_param_cnt
        node_cnt += input_node_cnt
    return param_cnt, node_cnt


def render_summary(network: HeadlessTransformer, output_node: Node):
    env = Environment(loader=FileSystemLoader(template_directory))
    template = env.get_template("summary_template.html")

    graph_params, graph_nodes = graph_stats([output_node])
    net_params = sum(layer.num_params() for layer in network.layers)

    # Populate and render the template
    return template.render(
        n_layers=len(network.layers),
        net_params=net_params,
        d_model=network.d,
        d_head=network.d_head,
        graph_nodes=graph_nodes,
        graph_params=graph_params,
    )


def render_network(network: HeadlessTransformer, output_node: Node):
    env = Environment(loader=FileSystemLoader(template_directory))
    template = env.get_template("report_template.html")

    layers = []
    for i, l in enumerate(network.layers):
        layers.append(render_attn_layer(l.attn, i))
        layers.append(render_ffn_layer(l.ffn, i))

    summary = render_summary(network, output_node)

    # Populate and render the template
    return template.render(layers=layers, summary=summary)


def make_report(network: HeadlessTransformer, output_node: Node, report_name: str):
    # Generate the current timestamp with microseconds for higher granularity
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # Create a unique filename using the high-granularity timestamp
    file_name = f"/tmp/modelscript_compile_{report_name}_{current_time}.html"

    # Log the action
    print(f"Generating report and saving to {file_name}")

    # Render the network to HTML
    rendered_html = render_network(network, output_node)

    # Save the rendered HTML to the file
    try:
        with open(file_name, "w") as f:
            f.write(rendered_html)
    except Exception as e:
        print(f"An error occurred: {e}")
