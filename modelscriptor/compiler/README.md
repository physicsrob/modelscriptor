

# Examples of things getting compiled to FFN Sub Layer
## A non-compilable node (e.g. an InputNode)
A non-compilable node can be passed through to the previous layer via the skip connection.

Steps:
1. `FFNSubLayer.get_strategies` is called, which is delegated to `get_sequential_placement_strategies`
1. `get_sequential_placement_strategies` finds strategies for the first component, the skip connection
    1. The skip connection returns two strategies. Both strategies are adding a zero-constant and the input node, but there are two relevant orderings.
    1. The inputs to the strategy are calculated with `s.get_component_inputs(current_component)`, note the skip node is not considered an input, which means that it doesn't get compiled in the next call. This is quite hacky.
1. `get_sequential_placement_strategies` is called recursively with the inputs to the skip component.
   1. The next component is the linear layer.
      1. If the zero constant goes to the linear node, it should be compilable.
      1. If the InputNode goes to the linear node, it s not compilable.
1. `get_sequential_placement_strategies` is called recursively with the input to the linear layer.
   1. If the linear layer got the constant, it has a placeholder input.  This should result in a NoOPStrategy.