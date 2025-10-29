//! Here we define the core language AST

use crate::lexer::{CoreType, Identifier};

// Top-level statements and declarations
#[derive(Debug, Clone)]
pub enum TopLevel {
    Definition(Definition),
    Import(ImportStatement),
}

// Import and namespace management
#[derive(Debug, Clone)]
pub struct ImportStatement {
    pub file: Identifier,
}

// Definitions
#[derive(Debug, Clone)]
pub enum Definition {
    Function(FunctionDef),
    TypeAlias(TypeAlias),
    Struct(StructDef),
    Impl(ImplBlock),
    GlobalDef(GlobalDef),
}

#[derive(Debug, Clone)]
pub struct FunctionDef {
    pub name: Identifier,
    pub params: Vec<Parameter>,
    pub return_type: TypeExpr,
    pub body: Block,
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: Identifier,
    pub type_expr: TypeExpr,
}

#[derive(Debug, Clone)]
pub struct TypeAlias {
    pub name: Identifier,
    pub type_expr: TypeExpr,
}

#[derive(Debug, Clone)]
pub struct StructDef {
    pub name: Identifier,
    pub fields: Vec<FieldDef>,
}

#[derive(Debug, Clone)]
pub struct FieldDef {
    pub name: Identifier,
    pub type_expr: TypeExpr,
}

#[derive(Debug, Clone)]
pub struct ImplBlock {
    pub struct_name: Identifier,
    pub methods: Vec<FunctionDef>,
}

#[derive(Debug, Clone)]
pub struct GlobalDef {
    pub name: Identifier,
    pub type_expr: TypeExpr,
    pub value: Expr,
}

#[derive(Debug, Clone)]
pub enum TypeExpr {
    Core(CoreType),
    Function(Vec<TypeExpr>, Box<TypeExpr>),
    List(Box<TypeExpr>),
    Tuple(Vec<TypeExpr>),
    HashMap(Box<TypeExpr>, Box<TypeExpr>),
    NamedType(Identifier),
}

// Literals
#[derive(Debug, Clone)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    Char(char),
    String(String),
    Bool(bool),
    Unit,
}

// Expressions
#[derive(Debug, Clone)]
pub enum Expr {
    Literal(Literal),
    Identifier(Identifier),
    Lambda(LambdaExpr),
    Tuple(Vec<Expr>),
    List(Vec<Expr>),
    Map(Vec<(Expr, Expr)>),
    MemberAccess(Box<Expr>, Identifier),
    FunctionCall(Box<Expr>, Vec<Expr>),
    MethodCall(Box<Expr>, Identifier, Vec<Expr>),
    UnaryOp(UnaryOperator, Box<Expr>),
    BinaryOp(BinaryOperator, Box<Expr>, Box<Expr>),
    Comparison(Comparison, Box<Expr>, Box<Expr>),
    Cast(Box<Expr>, TypeExpr),
    Block(Block),
    If(IfExpr),
    Loop(LoopExpr),
}

#[derive(Debug, Clone)]
pub enum LValue {
    Identifier(Identifier),
    MemberAccess(Box<LValue>, Identifier),
    TupleAccess(Box<LValue>, usize),
    ListAccess(Box<LValue>, Expr),
    MapAccess(Box<LValue>, Expr),
}

#[derive(Debug, Clone)]
pub struct LambdaExpr {
    pub params: Vec<Parameter>,
    pub body: Block,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub enum Statement {
    Break,
    Continue,
    Return(Option<Box<Expr>>),
    Assignment(LValue, Box<Expr>),
    Expr(Box<Expr>),
    Let(LetStatement),
}

#[derive(Debug, Clone)]
pub struct LetStatement {
    pub name: Identifier,
    pub mutable: bool,
    pub type_expr: Option<TypeExpr>,
    pub value: Box<Expr>,
}

// Control flow
#[derive(Debug, Clone)]
pub struct IfExpr {
    pub condition: Box<Expr>,
    pub then_block: Block,
    pub elif_branches: Vec<(Expr, Block)>,
    pub else_block: Option<Block>,
}

#[derive(Debug, Clone)]
pub enum LoopExpr {
    While {
        condition: Box<Expr>,
        body: Block,
    },
    For {
        var: Identifier,
        iterable: Box<Expr>,
        body: Block,
    },
}

// Operators
#[derive(Debug, Clone)]
pub enum UnaryOperator {
    Neg,
    Not,
}

#[derive(Debug, Clone)]
pub enum BinaryOperator {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,

    // Boolean
    And,
    Or,

    // List operations
    Append,
}

#[derive(Debug, Clone)]
pub enum Comparison {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}
