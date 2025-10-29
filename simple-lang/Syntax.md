# Syntax

This file will be where I keep my syntax definitions

## Literals
* Integers: decimal `0`, hex `0xabc`, binary `0b10011`
<!-- TODO: * Floats: decimal `3.14`, scientific `1e-9` -->
* Chars: `'a'`
* String: `"Hello, world!"`
* Booleans: `true`, `false`
* Unit: `()`

## Lambda
* Anonymous function `|<args1> : <type1>, <arg2>: <type2>, ...| { ... }`

## Type aliases
* `type <name> = <type>;`

## Expressions
* Parentheses do nothing but group `(<expr>)`
* Tuples `(a, b, c)`
* Lists `[a,b,c]`
* Maps `{<key>: <value>, ...}`

## Idents
* Identifiers can be `\alpha[\alphanum|_]*`

## Whitespace
* Whitespace insensitive

## Each line
* Lines end with `;` (unless commented)

## Code block
* Code blocks are contained within `{ ... }`

## Comments
* Single line comments with the initial `//`

## Variables/Consts
* Mutable variables: `mut <name> : <type>? = <value>`
* Constant variables: `const <name> : <type>? = <value>`

## Loops
* For loops: `for (<varname> in <iterable>) { ... }`
* While loops: `while (<condition>) { ... }`
* Continue statements: `continue`
* Break statements: `break`

## Conditionals
* If-then-else `if (<conditional>) { ... } elif (<conditional) { ... } else { ... }`

## Function declarations
* Declaration: `def <name>(<binders>) -> <return type> { <body> }`
* Return statements: `return <value>`

## Structures
* Declaration `struct <name> { <fieldname> : <fieldtype>,* }`
* Member access `<name>.<field>`
* Impl declaration `impl <structname> { <methods> }`
* Impl calling `<struct>.<method>(...)`

## Globals
* Declaration `global <name> : <type> = <expr>`

## Unary operators

### Arithmetic
* Negative `-<num>`

### Boolean
* not `!`

## Binary operators

### Arithmetic
* Plus `+`
* Times `*`
* Subtract `-`
* Divide `/`

### Boolean ops
* and/or: `&&`, `||`

### Comparison
* eq/neq/lt/le/gt/ge: `==`, `!=`, `<`, `<=`, `>`, `>=`

### Lists
* Append `++`

## Casting
* `(<value> : <type>)`

## Import statements
* Import library `import <file>`
<!-- TODO: Reconsider this
* Namespace opening `open <namespace>`
* Namespace closing `close <namespace>`
* enter namespace `namespace <name> {}` -->
