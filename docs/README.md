# adjointly documentation

Focused pages, each standing on its own. They ship in the npm package, so an agent or a person
working offline has them beside the code. Every `ts` example on these pages is type-checked and run
by the test suite, and a line ending in `//=> value` is asserted.

| Page                       | Read it for                                                                                                                   |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| [concepts.md](concepts.md) | How reverse and forward mode work here, what the tape stores, and when to use which                                           |
| [api.md](api.md)           | Every export, grouped by topic, with its contract                                                                             |
| [recipes.md](recipes.md)   | Worked patterns: inner loops, shared prefixes, several outputs, Jacobians, parameters with constraints, testing your gradient |
| [pitfalls.md](pitfalls.md) | The mistakes that silently give a wrong derivative, and how to avoid them                                                     |

A worked, end-to-end example lives in
[examples/production-line/](../examples/production-line/README.md): maintenance scheduling on a
simulated production line, using the gradient to rank discrete moves and forward mode for a
parameter sensitivity.

The package's [README](../README.md) has the overview, and [llms.txt](../llms.txt) a compact
summary for language models.
