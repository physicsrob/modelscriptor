# Operaiton Naming Convention

Operation Naming Convention:
- Operation names start with a verb where possible. Most operations are performing actions on nodes, and verbs make it easier to interpret what the operation will do.
- Use the terms scalar and vector to avoid ambiguity.
- It can be assumed that the action being performed is one one or more node.  If, however, the action is being performed between a node and a vector, state so.  For instance `add_scalar` clearly indicates that the input node is being added with a scalar.
- Stick to PyTorch naming conventions where they align. If there's an analogous operation in PyTorch, its naming convention can serve as guidance, but only if it doesn't confuse the purpose of the function in this new context.
- Keep names as concise as possible without sacrificing clarity. If a name can be shorter without being less clear, it should be.
- Operations that are predicated on a boolean condition should be prefixed with `cond_`
- Operations that conditionally copy input to output should be prefixed with `select_`


Parameter naming convention:
- Boolean input nodes should always be named `cond`.
- If an operation acts on a single node input, that node should be named `inp`.
- If an operation acts on two node inputs, those nodes should be named `inp1` and `inp2`.
- If an operation has a scalar parameter, the parameter name should be scalar.
- If an operation has a vector parameter, the parameter name should be vector.
