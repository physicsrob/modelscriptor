def compile_simple(d: int, output_node: Node) -> SimpleNetwork:
    # Start with the first layer and try to compile as much as possible.

    net = SimpleNetwork(d, output_node)
    while True:
        layer = net.add_layer()
        for node in layer.out_nodes:
            strategy = sorted(
                layer.get_strategies(output_node), key=lambda s: -s.points
            )[0]
            layer.apply_strategy(strategy)
        # Woohoo -- layer is done!
        ...


def compile_med(d: int, output_node: Node) -> MedSimpleNetwork:
    # Start with the first layer and try to compile as much as possible.

    net = MedSimpleNetwork(d, output_node)
    while True:
        layer = net.add_layer()
        for node in layer.out_nodes:
            strategy = sorted(
                layer.get_strategies(output_node), key=lambda s: -s.points
            )[0]
            layer.apply_strategy(strategy)
        # Woohoo -- layer is done!
        ...
