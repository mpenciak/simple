//! Here we define the parser

use crate::{
    ast::{
        Block, Definition, Expr, FieldDef, FunctionDef, GlobalDef, ImplBlock, ImportStatement,
        LValue, LetStatement, Literal, Parameter, Statement, StructDef, TopLevel, TypeAlias,
        TypeExpr,
    },
    errors::{self, parser::Error},
    lexer::{CoreType, Delimiter, Identifier, Keyword, LiteralNumber, Punctuation, Token, Tokens},
};
use nom::{
    self,
    branch::alt,
    combinator::{all_consuming, opt},
    multi::{many0, separated_list0},
    sequence::preceded,
    Err::Error as NomError,
    IResult, Input, Parser,
};

type ParseResult<'a, T> = IResult<Tokens<'a>, T, errors::parser::Error<Tokens<'a>>>;

trait TokenExtract: Sized {
    fn extract(token: Token) -> Option<Self>;
}

macro_rules! impl_token_extract {
    ($ty:ty, $branch:ident) => {
        impl TokenExtract for $ty {
            fn extract(token: Token) -> Option<Self> {
                match token {
                    Token::$branch(value) => Some(value),
                    _ => None,
                }
            }
        }
    };
}

impl_token_extract!(Identifier, Identifier);
impl_token_extract!(Keyword, Keyword);
impl_token_extract!(Delimiter, Delimiter);
impl_token_extract!(CoreType, TypeWord);

macro_rules! parse_variant {
    ($ty:ty) => {
        parse_atom::<$ty>()
    };
}

pub fn parse_file(input: Tokens) -> ParseResult<Vec<TopLevel>> {
    all_consuming(many0(parse_top_level)).parse(input)
}

fn parse_top_level(input: Tokens) -> ParseResult<TopLevel> {
    alt((
        parse_definition.map(TopLevel::Definition),
        parse_import.map(TopLevel::Import),
    ))
    .parse(input)
}

fn parse_definition(input: Tokens) -> ParseResult<Definition> {
    alt((
        parse_func_def.map(Definition::Function),
        parse_type_alias.map(Definition::TypeAlias),
        parse_struct_def.map(Definition::Struct),
        parse_impl.map(Definition::Impl),
        parse_global.map(Definition::GlobalDef),
    ))
    .parse(input)
}

fn parse_func_def(input: Tokens) -> ParseResult<FunctionDef> {
    let (input, (_, name, params, _, return_type, body)) = (
        expect_keywords(vec![Keyword::Def]),
        parse_identifier,
        surrounded_by(
            Delimiter::LeftParen,
            |tokens| {
                separated_list0(
                    expect_atomic_token(Token::Punctuation(Punctuation::Comma)),
                    parse_parameter,
                )
                .parse(tokens)
            },
            Delimiter::RightParen,
        ),
        expect_atomic_token(Token::Punctuation(Punctuation::Arrow)),
        parse_type_expr,
        parse_block,
    )
        .parse(input)?;

    let func_def = FunctionDef {
        name,
        params,
        return_type,
        body,
    };

    Ok((input, func_def))
}

fn parse_parameter(input: Tokens) -> ParseResult<Parameter> {
    let (input, (name, _, type_expr)) = (
        parse_identifier,
        expect_atomic_token(Token::Punctuation(Punctuation::Colon)),
        parse_type_expr,
    )
        .parse(input)?;

    let param = Parameter { name, type_expr };

    Ok((input, param))
}

fn parse_block(input: Tokens) -> ParseResult<Block> {
    separated_list0(
        expect_atomic_token(Token::Punctuation(Punctuation::Semicolon)),
        parse_statement,
    )
    .map(|statements| Block { statements })
    .parse(input)
}

fn parse_statement(input: Tokens) -> ParseResult<Statement> {
    alt((
        parse_break,
        parse_continue,
        parse_return.map(|expr| Statement::Return(expr.map(Box::new))),
        parse_assignment.map(|(lvalue, expr)| Statement::Assignment(lvalue, Box::new(expr))),
        parse_let.map(Statement::Let),
        parse_expr.map(|expr| Statement::Expr(Box::new(expr))),
    ))
    .parse(input)
}

fn parse_break(input: Tokens) -> ParseResult<Statement> {
    let (input, _) = expect_keywords(vec![Keyword::Break]).parse(input)?;

    Ok((input, Statement::Break))
}

fn parse_continue(input: Tokens) -> ParseResult<Statement> {
    let (input, _) = expect_keywords(vec![Keyword::Continue]).parse(input)?;

    Ok((input, Statement::Continue))
}

fn parse_return(input: Tokens) -> ParseResult<Option<Expr>> {
    let (input, _) = expect_keywords(vec![Keyword::Return]).parse(input)?;
    let (input, maybe_expr) = opt(parse_expr).parse(input)?;

    Ok((input, maybe_expr))
}

fn parse_assignment(input: Tokens) -> ParseResult<(LValue, Expr)> {
    let (input, (lvalue, _, expr)) =
        (parse_lvalue, expect_atomic_token(Token::Assign), parse_expr).parse(input)?;

    Ok((input, (lvalue, expr)))
}

fn parse_lvalue(input: Tokens) -> ParseResult<LValue> {
    let (input, lval) = surrounded_by(
        Delimiter::LeftParen,
        |tokens| {
            alt((
                // Identifier LVal
                parse_identifier.map(|ident| LValue::Identifier(ident)),
                // Member access LVal
                (
                    parse_lvalue,
                    expect_atomic_token(Token::Punctuation(Punctuation::Dot)),
                    parse_identifier,
                )
                    .map(|(lval, _, member)| LValue::MemberAccess(Box::new(lval), member)),
                // Tuple access LVal
                (
                    parse_lvalue,
                    expect_atomic_token(Token::Punctuation(Punctuation::Dot)),
                    parse_dec_num_lit,
                )
                    .map(|(lval, _, num)| LValue::TupleAccess(Box::new(lval), num)),
                // List access
                (
                    parse_lvalue,
                    surrounded_by(Delimiter::LeftBracket, parse_expr, Delimiter::RightBracket),
                )
                    .map(|(lval, expr)| LValue::ListAccess(Box::new(lval), expr)),
                // Map access
                (
                    parse_lvalue,
                    surrounded_by(
                        Delimiter::LeftBrace,
                        surrounded_by(Delimiter::LeftBracket, parse_expr, Delimiter::RightBracket),
                        Delimiter::RightBrace,
                    ),
                )
                    .map(|(lval, expr)| LValue::MapAccess(Box::new(lval), expr)),
            ))
            .parse(tokens)
        },
        Delimiter::RightParen,
    )
    .parse(input)?;

    Ok((input, lval))
}

fn parse_let(input: Tokens) -> ParseResult<LetStatement> {
    let (input, (_, name, mutable, type_expr, _, value)) = (
        expect_keywords(vec![Keyword::Let]),
        parse_identifier,
        opt(expect_keywords(vec![Keyword::Mut])),
        opt(preceded(
            expect_atomic_token(Token::Punctuation(Punctuation::Colon)),
            parse_type_expr,
        )),
        expect_atomic_token(Token::Assign),
        parse_expr,
    )
        .parse(input)?;

    let let_stmt = LetStatement {
        name,
        mutable: mutable.is_some(),
        type_expr,
        value: Box::new(value),
    };

    Ok((input, let_stmt))
}

fn parse_type_alias(input: Tokens) -> ParseResult<TypeAlias> {
    let (input, (_, name, _, type_expr)) = (
        expect_keywords(vec![Keyword::Type]),
        parse_identifier,
        expect_atomic_token(Token::Assign),
        parse_type_expr,
    )
        .parse(input)?;

    let alias = TypeAlias { name, type_expr };

    Ok((input, alias))
}

fn parse_struct_def(input: Tokens) -> ParseResult<StructDef> {
    let (input, (_, name, fields)) = (
        expect_keywords(vec![Keyword::Struct]),
        parse_identifier,
        surrounded_by(
            Delimiter::LeftBrace,
            |tokens| many0(parse_struct_field).parse(tokens),
            Delimiter::RightBrace,
        ),
    )
        .parse(input)?;

    let struct_def = StructDef { name, fields };

    Ok((input, struct_def))
}

fn parse_struct_field(input: Tokens) -> ParseResult<FieldDef> {
    let (input, (name, _, type_expr, _)) = (
        parse_identifier,
        expect_atomic_token(Token::Punctuation(Punctuation::Colon)),
        parse_type_expr,
        expect_atomic_token(Token::Punctuation(Punctuation::Comma)),
    )
        .parse(input)?;

    let field_def = FieldDef { name, type_expr };

    Ok((input, field_def))
}

fn parse_impl(input: Tokens) -> ParseResult<ImplBlock> {
    let (input, (_, struct_name, methods)) = (
        expect_keywords(vec![Keyword::Impl]),
        parse_identifier,
        surrounded_by(
            Delimiter::LeftBrace,
            |tokens| many0(parse_func_def).parse(tokens),
            Delimiter::RightBrace,
        ),
    )
        .parse(input)?;

    let impl_block = ImplBlock {
        struct_name,
        methods,
    };

    Ok((input, impl_block))
}

fn parse_global(input: Tokens) -> ParseResult<GlobalDef> {
    let (input, (_, name, _, type_expr, _, expr)) = (
        expect_keywords(vec![Keyword::Global]),
        parse_identifier,
        expect_atomic_token(Token::Punctuation(Punctuation::Colon)),
        parse_type_expr,
        expect_atomic_token(Token::Assign),
        parse_expr,
    )
        .parse(input)?;

    let global_def = GlobalDef {
        name,
        type_expr: type_expr,
        value: expr,
    };

    Ok((input, global_def))
}

fn parse_type_expr(input: Tokens) -> ParseResult<TypeExpr> {
    alt((
        // They can be surrounded by parens
        surrounded_by(Delimiter::LeftParen, parse_type_expr, Delimiter::RightParen),
        // Core types
        parse_core_type.map(|core| TypeExpr::Core(core)),
        // Function types
        parse_lambda_type,
        // Lists
        (
            expect_keywords(vec![Keyword::List]),
            surrounded_by(
                Delimiter::LeftBracket,
                parse_type_expr,
                Delimiter::RightBracket,
            ),
        )
            .map(|(_, type_expr)| TypeExpr::List(Box::new(type_expr))),
        // Tuples
        (
            expect_keywords(vec![Keyword::Tuple]),
            surrounded_by(
                Delimiter::LeftBracket,
                |tokens| {
                    separated_list0(
                        expect_atomic_token(Token::Punctuation(Punctuation::Comma)),
                        parse_type_expr,
                    )
                    .parse(tokens)
                },
                Delimiter::RightBracket,
            ),
        )
            .map(|(_, type_expr)| TypeExpr::Tuple(type_expr)),
        // Maps
        (
            expect_keywords(vec![Keyword::Map]),
            surrounded_by(
                Delimiter::LeftBracket,
                |tokens| {
                    (
                        parse_type_expr,
                        expect_atomic_token(Token::Punctuation(Punctuation::Colon)),
                        parse_type_expr,
                    )
                        .parse(tokens)
                },
                Delimiter::RightBracket,
            ),
        )
            .map(|(_, (keys, _, vals))| TypeExpr::HashMap(Box::new(keys), Box::new(vals))),
        parse_identifier.map(|ident| TypeExpr::NamedType(ident)),
    ))
    .parse(input)
}

fn parse_core_type(input: Tokens) -> ParseResult<CoreType> {
    parse_variant!(CoreType).parse(input)
}

fn parse_lambda_type(input: Tokens) -> ParseResult<TypeExpr> {
    let (input, (_, arg_types, _, out_type)) = (
        expect_keywords(vec![Keyword::Fun]),
        surrounded_by(
            Delimiter::LeftBracket,
            |tokens| {
                separated_list0(
                    expect_atomic_token(Token::Punctuation(Punctuation::Comma)),
                    parse_type_expr,
                )
                .parse(tokens)
            },
            Delimiter::RightBracket,
        ),
        expect_atomic_token(Token::Punctuation(Punctuation::Arrow)),
        parse_type_expr,
    )
        .parse(input)?;

    Ok((input, TypeExpr::Function(arg_types, Box::new(out_type))))
}

fn parse_import(input: Tokens) -> ParseResult<ImportStatement> {
    let (input, filename) =
        preceded(expect_keywords(vec![Keyword::Import]), parse_identifier).parse(input)?;
    Ok((input, ImportStatement { file: filename }))
}

// TODO: Finish this
// ========================= EXPRESSIONS ===============================

// Entry point for expression parsing with precedence
fn parse_expr(input: Tokens) -> ParseResult<Expr> {
    parse_or_expr(input)
}

// Precedence levels (lowest to highest):
// || -> && -> comparison -> + - ++ -> * / -> unary -> postfix -> primary

fn parse_or_expr(input: Tokens) -> ParseResult<Expr> {
    let (mut rest, mut lhs) = parse_and_expr(input)?;
    loop {
        if let Some(Token::BinaryOp(binop)) = peek_token(rest.clone()) {
            if matches!(binop, crate::lexer::BinaryOperator::Or) {
                let (after_op, _) = consume_one(rest)?;
                let (after_rhs, rhs) = parse_and_expr(after_op)?;
                lhs = Expr::BinaryOp(crate::ast::BinaryOperator::Or, Box::new(lhs), Box::new(rhs));
                rest = after_rhs;
                continue;
            }
        }
        break;
    }
    Ok((rest, lhs))
}

fn parse_and_expr(input: Tokens) -> ParseResult<Expr> {
    let (mut rest, mut lhs) = parse_comparison_expr(input)?;
    loop {
        if let Some(Token::BinaryOp(binop)) = peek_token(rest.clone()) {
            if matches!(binop, crate::lexer::BinaryOperator::And) {
                let (after_op, _) = consume_one(rest)?;
                let (after_rhs, rhs) = parse_comparison_expr(after_op)?;
                lhs = Expr::BinaryOp(
                    crate::ast::BinaryOperator::And,
                    Box::new(lhs),
                    Box::new(rhs),
                );
                rest = after_rhs;
                continue;
            }
        }
        break;
    }
    Ok((rest, lhs))
}

fn parse_comparison_expr(input: Tokens) -> ParseResult<Expr> {
    let (mut rest, mut lhs) = parse_add_expr(input)?;
    if let Some(Token::ComparisonOp(cmp)) = peek_token(rest.clone()) {
        let (after_op, _) = consume_one(rest)?;
        let (after_rhs, rhs) = parse_add_expr(after_op)?;
        let cmp_ast = match cmp {
            crate::lexer::ComparisonOperator::Eq => crate::ast::Comparison::Eq,
            crate::lexer::ComparisonOperator::Ne => crate::ast::Comparison::Ne,
            crate::lexer::ComparisonOperator::Lt => crate::ast::Comparison::Lt,
            crate::lexer::ComparisonOperator::Le => crate::ast::Comparison::Le,
            crate::lexer::ComparisonOperator::Gt => crate::ast::Comparison::Gt,
            crate::lexer::ComparisonOperator::Ge => crate::ast::Comparison::Ge,
        };
        lhs = Expr::Comparison(cmp_ast, Box::new(lhs), Box::new(rhs));
        rest = after_rhs;
    }
    Ok((rest, lhs))
}

fn parse_add_expr(input: Tokens) -> ParseResult<Expr> {
    let (mut rest, mut lhs) = parse_mul_expr(input)?;
    loop {
        match peek_token(rest.clone()) {
            Some(Token::BinaryOp(op @ crate::lexer::BinaryOperator::Add))
            | Some(Token::BinaryOp(op @ crate::lexer::BinaryOperator::Sub))
            | Some(Token::BinaryOp(op @ crate::lexer::BinaryOperator::Append)) => {
                let (after_op, token) = consume_one(rest)?;
                let (after_rhs, rhs) = parse_mul_expr(after_op)?;
                let op = match token {
                    Token::BinaryOp(crate::lexer::BinaryOperator::Add) => {
                        crate::ast::BinaryOperator::Add
                    }
                    Token::BinaryOp(crate::lexer::BinaryOperator::Sub) => {
                        crate::ast::BinaryOperator::Sub
                    }
                    Token::BinaryOp(crate::lexer::BinaryOperator::Append) => {
                        crate::ast::BinaryOperator::Append
                    }
                    _ => unreachable!(),
                };
                lhs = Expr::BinaryOp(op, Box::new(lhs), Box::new(rhs));
                rest = after_rhs;
            }
            _ => break,
        }
    }
    Ok((rest, lhs))
}

fn parse_mul_expr(input: Tokens) -> ParseResult<Expr> {
    let (mut rest, mut lhs) = parse_unary_expr(input)?;
    loop {
        match peek_token(rest.clone()) {
            Some(Token::BinaryOp(crate::lexer::BinaryOperator::Mul))
            | Some(Token::BinaryOp(crate::lexer::BinaryOperator::Div)) => {
                let (after_op, token) = consume_one(rest)?;
                let (after_rhs, rhs) = parse_unary_expr(after_op)?;
                let op = match token {
                    Token::BinaryOp(crate::lexer::BinaryOperator::Mul) => {
                        crate::ast::BinaryOperator::Mul
                    }
                    Token::BinaryOp(crate::lexer::BinaryOperator::Div) => {
                        crate::ast::BinaryOperator::Div
                    }
                    _ => unreachable!(),
                };
                lhs = Expr::BinaryOp(op, Box::new(lhs), Box::new(rhs));
                rest = after_rhs;
            }
            _ => break,
        }
    }
    Ok((rest, lhs))
}

fn parse_unary_expr(input: Tokens) -> ParseResult<Expr> {
    match peek_token(input.clone()) {
        Some(Token::UnaryOp(op)) => {
            let (rest, token) = consume_one(input)?;
            let (rest, expr) = parse_unary_expr(rest)?; // allow chaining
            let uop = match token {
                Token::UnaryOp(crate::lexer::UnaryOperator::Negative) => {
                    crate::ast::UnaryOperator::Neg
                }
                Token::UnaryOp(crate::lexer::UnaryOperator::Not) => crate::ast::UnaryOperator::Not,
                _ => unreachable!(),
            };
            Ok((rest, Expr::UnaryOp(uop, Box::new(expr))))
        }
        _ => parse_postfix_expr(input),
    }
}

fn parse_postfix_expr(input: Tokens) -> ParseResult<Expr> {
    let (mut rest, mut expr) = parse_primary(input)?;
    loop {
        match peek_token(rest.clone()) {
            Some(Token::Delimiter(Delimiter::LeftParen)) => {
                // function call
                let (after_lpar, _) = consume_one(rest)?;
                let (after_args, args) = parse_argument_list(after_lpar)?;
                expr = Expr::FunctionCall(Box::new(expr), args);
                rest = after_args;
            }
            Some(Token::Punctuation(Punctuation::Dot)) => {
                let (after_dot, _) = consume_one(rest)?;
                // expect identifier
                if let Some(Token::Identifier(name)) = peek_token(after_dot.clone()) {
                    let (after_ident, ident_token) = consume_one(after_dot)?;
                    let ident = match ident_token {
                        Token::Identifier(s) => s,
                        _ => unreachable!(),
                    };
                    // method call or member access?
                    if let Some(Token::Delimiter(Delimiter::LeftParen)) =
                        peek_token(after_ident.clone())
                    {
                        let (after_lpar, _) = consume_one(after_ident)?;
                        let (after_args, args) = parse_argument_list(after_lpar)?;
                        expr = Expr::MethodCall(Box::new(expr), ident, args);
                        rest = after_args;
                    } else {
                        expr = Expr::MemberAccess(Box::new(expr), ident);
                        rest = after_ident;
                    }
                } else {
                    return Err(NomError(Error::tag(after_dot)));
                }
            }
            _ => break,
        }
    }
    Ok((rest, expr))
}

fn parse_argument_list(input: Tokens) -> ParseResult<Vec<Expr>> {
    // already consumed '('
    let mut rest = input;
    let mut args = Vec::new();
    if let Some(Token::Delimiter(Delimiter::RightParen)) = peek_token(rest.clone()) {
        let (after_rpar, _) = consume_one(rest)?;
        return Ok((after_rpar, args));
    }
    loop {
        let (after_expr, arg) = parse_expr(rest)?;
        args.push(arg);
        if let Some(Token::Punctuation(Punctuation::Comma)) = peek_token(after_expr.clone()) {
            let (after_comma, _) = consume_one(after_expr)?;
            rest = after_comma;
            continue;
        } else if let Some(Token::Delimiter(Delimiter::RightParen)) = peek_token(after_expr.clone())
        {
            let (after_rpar, _) = consume_one(after_expr)?;
            rest = after_rpar;
            break;
        } else {
            return Err(NomError(Error::tag(after_expr)));
        }
    }
    Ok((rest, args))
}

fn parse_primary(input: Tokens) -> ParseResult<Expr> {
    match peek_token(input.clone()) {
        Some(Token::Identifier(_)) => {
            let (rest, tok) = consume_one(input)?;
            if let Token::Identifier(name) = tok {
                Ok((rest, Expr::Identifier(name)))
            } else {
                unreachable!()
            }
        }
        Some(Token::Number(_))
        | Some(Token::Char(_))
        | Some(Token::True)
        | Some(Token::False)
        | Some(Token::Unit)
        | Some(Token::String(_)) => parse_literal(input).map(|(r, l)| (r, Expr::Literal(l))),
        Some(Token::Delimiter(Delimiter::LeftParen)) => parse_paren_expr(input),
        Some(Token::Delimiter(Delimiter::LeftBracket)) => parse_list_expr(input),
        Some(Token::Delimiter(Delimiter::LeftBrace)) => parse_block_or_map_expr(input),
        Some(Token::Punctuation(Punctuation::Pipe)) => parse_lambda_expr(input),
        Some(Token::Keyword(Keyword::If)) => parse_if_expr(input),
        Some(Token::Keyword(Keyword::While)) | Some(Token::Keyword(Keyword::For)) => {
            parse_loop_expr(input)
        }
        _ => Err(NomError(Error::tag(input))),
    }
}

fn parse_paren_expr(input: Tokens) -> ParseResult<Expr> {
    let (after_lpar, _) = consume_expected_delimiter(input, Delimiter::LeftParen)?;
    // Empty parens already handled as Unit literal token, so here must contain something
    let (rest_after_first, first_expr) = parse_expr(after_lpar.clone())?;
    // Cast?
    if let Some(Token::Punctuation(Punctuation::Colon)) = peek_token(rest_after_first.clone()) {
        let (after_colon, _) = consume_one(rest_after_first)?;
        let (after_type, ty) = parse_type_expr(after_colon)?;
        let (after_rpar, _) = consume_expected_delimiter(after_type, Delimiter::RightParen)?;
        return Ok((after_rpar, Expr::Cast(Box::new(first_expr), ty)));
    }
    // Tuple or grouped
    let mut exprs = vec![first_expr];
    let mut rest = rest_after_first;
    while let Some(Token::Punctuation(Punctuation::Comma)) = peek_token(rest.clone()) {
        let (after_comma, _) = consume_one(rest)?;
        let (after_expr, next_expr) = parse_expr(after_comma)?;
        exprs.push(next_expr);
        rest = after_expr;
    }
    let (after_rpar, _) = consume_expected_delimiter(rest, Delimiter::RightParen)?;
    if exprs.len() == 1 {
        Ok((after_rpar, exprs.into_iter().next().unwrap()))
    } else {
        Ok((after_rpar, Expr::Tuple(exprs)))
    }
}

fn parse_list_expr(input: Tokens) -> ParseResult<Expr> {
    let (mut rest, _) = consume_expected_delimiter(input, Delimiter::LeftBracket)?;
    let mut elems = Vec::new();
    if let Some(Token::Delimiter(Delimiter::RightBracket)) = peek_token(rest.clone()) {
        let (after, _) = consume_one(rest)?; // consume ']'
        return Ok((after, Expr::List(elems)));
    }
    loop {
        let (after_elem, elem) = parse_expr(rest)?;
        elems.push(elem);
        if let Some(Token::Punctuation(Punctuation::Comma)) = peek_token(after_elem.clone()) {
            let (after_comma, _) = consume_one(after_elem)?;
            rest = after_comma;
            continue;
        } else if let Some(Token::Delimiter(Delimiter::RightBracket)) =
            peek_token(after_elem.clone())
        {
            let (after_rbr, _) = consume_one(after_elem)?;
            rest = after_rbr;
            break;
        } else {
            return Err(NomError(Error::tag(after_elem)));
        }
    }
    Ok((rest, Expr::List(elems)))
}

fn parse_block_or_map_expr(input: Tokens) -> ParseResult<Expr> {
    // Lookahead to decide map vs block; simple heuristic: presence of ':' before first ';' or '}' indicates map
    if is_map_like(input.clone()) {
        parse_map_expr(input)
    } else {
        parse_block_expr(input)
    }
}

fn is_map_like(tokens: Tokens) -> bool {
    let mut iter = tokens.iter_elements();
    if let Some(Token::Delimiter(Delimiter::LeftBrace)) = iter.next() { /* good */
    } else {
        return false;
    }
    let mut depth = 0usize; // only track braces at top level of map literal
    for tok in iter {
        match tok {
            Token::Delimiter(Delimiter::LeftBrace) => depth += 1,
            Token::Delimiter(Delimiter::RightBrace) => {
                if depth == 0 {
                    return false;
                } else {
                    depth -= 1;
                }
            }
            Token::Punctuation(Punctuation::Colon) if depth == 0 => return true,
            Token::Punctuation(Punctuation::Semicolon) if depth == 0 => return false,
            _ => {}
        }
    }
    false
}

fn parse_map_expr(input: Tokens) -> ParseResult<Expr> {
    let (mut rest, _) = consume_expected_delimiter(input, Delimiter::LeftBrace)?;
    let mut entries = Vec::new();
    if let Some(Token::Delimiter(Delimiter::RightBrace)) = peek_token(rest.clone()) {
        let (after, _) = consume_one(rest)?; // empty map
        return Ok((after, Expr::Map(entries)));
    }
    loop {
        let (after_key, key) = parse_expr(rest)?;
        let (after_colon, _) = match peek_token(after_key.clone()) {
            Some(Token::Punctuation(Punctuation::Colon)) => consume_one(after_key)?,
            _ => return Err(NomError(Error::tag(after_key))),
        };
        let (after_val, val) = parse_expr(after_colon)?;
        entries.push((key, val));
        if let Some(Token::Punctuation(Punctuation::Comma)) = peek_token(after_val.clone()) {
            let (after_comma, _) = consume_one(after_val)?;
            rest = after_comma;
            continue;
        } else if let Some(Token::Delimiter(Delimiter::RightBrace)) = peek_token(after_val.clone())
        {
            let (after_rbrace, _) = consume_one(after_val)?;
            rest = after_rbrace;
            break;
        } else {
            return Err(NomError(Error::tag(after_val)));
        }
    }
    Ok((rest, Expr::Map(entries)))
}

fn parse_block_expr(input: Tokens) -> ParseResult<Expr> {
    let (mut rest, _) = consume_expected_delimiter(input, Delimiter::LeftBrace)?;
    let mut statements = Vec::new();
    // Empty block
    if let Some(Token::Delimiter(Delimiter::RightBrace)) = peek_token(rest.clone()) {
        let (after, _) = consume_one(rest)?;
        return Ok((after, Expr::Block(Block { statements })));
    }
    loop {
        let (after_stmt, stmt) = parse_statement(rest)?;
        statements.push(stmt);
        if let Some(Token::Punctuation(Punctuation::Semicolon)) = peek_token(after_stmt.clone()) {
            let (after_semi, _) = consume_one(after_stmt)?;
            rest = after_semi;
            continue;
        } else if let Some(Token::Delimiter(Delimiter::RightBrace)) = peek_token(after_stmt.clone())
        {
            let (after_rbrace, _) = consume_one(after_stmt)?;
            rest = after_rbrace;
            break;
        } else {
            // Missing closing brace
            return Err(NomError(Error::tag(after_stmt)));
        }
    }
    Ok((rest, Expr::Block(Block { statements })))
}

fn parse_lambda_expr(input: Tokens) -> ParseResult<Expr> {
    // |param: Type, ...| { body }
    let (mut rest, _) = consume_expected_punctuation(input, Punctuation::Pipe)?;
    let mut params = Vec::new();
    if let Some(Token::Punctuation(Punctuation::Pipe)) = peek_token(rest.clone()) {
        // no params
        let (after_pipe, _) = consume_one(rest)?;
        rest = after_pipe;
    } else {
        loop {
            // param name
            let (after_ident, ident) = match consume_identifier(rest) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            let (after_colon, _) = consume_expected_punctuation(after_ident, Punctuation::Colon)?;
            let (after_type, ty) = parse_type_expr(after_colon)?;
            params.push(Parameter {
                name: ident,
                type_expr: ty,
            });
            if let Some(Token::Punctuation(Punctuation::Comma)) = peek_token(after_type.clone()) {
                let (after_comma, _) = consume_one(after_type)?;
                rest = after_comma;
                continue;
            } else if let Some(Token::Punctuation(Punctuation::Pipe)) =
                peek_token(after_type.clone())
            {
                let (after_pipe, _) = consume_one(after_type)?;
                rest = after_pipe;
                break;
            } else {
                return Err(NomError(Error::tag(after_type)));
            }
        }
    }
    // body block
    let (after_body, body_expr) = parse_block_expr(rest)?;
    if let Expr::Block(body_block) = body_expr {
        Ok((
            after_body,
            Expr::Lambda(crate::ast::LambdaExpr {
                params,
                body: body_block,
            }),
        ))
    } else {
        Err(NomError(Error::tag(after_body)))
    }
}

fn parse_if_expr(input: Tokens) -> ParseResult<Expr> {
    // if (cond) { then } [elif (cond) { .. }]* [else { .. }]
    let (rest, _) = expect_keywords(vec![Keyword::If])(input)?; // reuse existing helper
    let (rest, _) = consume_expected_delimiter(rest, Delimiter::LeftParen)?;
    let (rest, cond) = parse_expr(rest)?;
    let (rest, _) = consume_expected_delimiter(rest, Delimiter::RightParen)?;
    let (rest, then_block_expr) = parse_block_expr(rest)?;
    let then_block = if let Expr::Block(b) = then_block_expr {
        b
    } else {
        return Err(NomError(Error::tag(rest)));
    };
    let mut elifs = Vec::new();
    let mut cur = rest;
    loop {
        if let Some(Token::Keyword(Keyword::Elif)) = peek_token(cur.clone()) {
            let (after_kw, _) = consume_one(cur)?;
            let (after_lpar, _) = consume_expected_delimiter(after_kw, Delimiter::LeftParen)?;
            let (after_cond, econd) = parse_expr(after_lpar)?;
            let (after_rpar, _) = consume_expected_delimiter(after_cond, Delimiter::RightParen)?;
            let (after_block_expr, blk_expr) = parse_block_expr(after_rpar)?;
            let blk = if let Expr::Block(b) = blk_expr {
                b
            } else {
                return Err(NomError(Error::tag(after_block_expr)));
            };
            elifs.push((econd, blk));
            cur = after_block_expr;
            continue;
        }
        break;
    }
    let mut else_block = None;
    if let Some(Token::Keyword(Keyword::Else)) = peek_token(cur.clone()) {
        let (after_else, _) = consume_one(cur)?;
        let (after_block_expr, blk_expr) = parse_block_expr(after_else)?;
        let blk = if let Expr::Block(b) = blk_expr {
            b
        } else {
            return Err(NomError(Error::tag(after_block_expr)));
        };
        else_block = Some(blk);
        cur = after_block_expr;
    }
    Ok((
        cur,
        Expr::If(crate::ast::IfExpr {
            condition: Box::new(cond),
            then_block,
            elif_branches: elifs,
            else_block,
        }),
    ))
}

fn parse_loop_expr(input: Tokens) -> ParseResult<Expr> {
    match peek_token(input.clone()) {
        Some(Token::Keyword(Keyword::While)) => {
            let (rest, _) = consume_one(input)?;
            let (rest, _) = consume_expected_delimiter(rest, Delimiter::LeftParen)?;
            let (rest, cond) = parse_expr(rest)?;
            let (rest, _) = consume_expected_delimiter(rest, Delimiter::RightParen)?;
            let (rest, body_expr) = parse_block_expr(rest)?;
            let body = if let Expr::Block(b) = body_expr {
                b
            } else {
                return Err(NomError(Error::tag(rest)));
            };
            Ok((
                rest,
                Expr::Loop(crate::ast::LoopExpr::While {
                    condition: Box::new(cond),
                    body,
                }),
            ))
        }
        Some(Token::Keyword(Keyword::For)) => {
            let (rest, _) = consume_one(input)?;
            let (rest, _) = consume_expected_delimiter(rest, Delimiter::LeftParen)?;
            let (rest, var) = match consume_identifier(rest) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            let (rest, _) = expect_keywords(vec![Keyword::In])(rest)?;
            let (rest, iterable) = parse_expr(rest)?;
            let (rest, _) = consume_expected_delimiter(rest, Delimiter::RightParen)?;
            let (rest, body_expr) = parse_block_expr(rest)?;
            let body = if let Expr::Block(b) = body_expr {
                b
            } else {
                return Err(NomError(Error::tag(rest)));
            };
            Ok((
                rest,
                Expr::Loop(crate::ast::LoopExpr::For {
                    var,
                    iterable: Box::new(iterable),
                    body,
                }),
            ))
        }
        _ => Err(NomError(Error::tag(input))),
    }
}

fn parse_literal(input: Tokens) -> ParseResult<Literal> {
    match peek_token(input.clone()) {
        Some(Token::Number(_)) => {
            let (rest, tok) = consume_one(input)?;
            if let Token::Number(lit) = tok {
                let value = match lit {
                    LiteralNumber::DecInt(v)
                    | LiteralNumber::HexInt(v)
                    | LiteralNumber::BinInt(v) => v,
                };
                Ok((rest, Literal::Integer(value)))
            } else {
                unreachable!()
            }
        }
        Some(Token::Char(_)) => {
            let (rest, tok) = consume_one(input)?;
            if let Token::Char(c) = tok {
                Ok((rest, Literal::Char(c)))
            } else {
                unreachable!()
            }
        }
        Some(Token::String(_)) => {
            let (rest, tok) = consume_one(input)?;
            if let Token::String(s) = tok {
                Ok((rest, Literal::String(s)))
            } else {
                unreachable!()
            }
        }
        Some(Token::True) => {
            let (rest, _) = consume_one(input)?;
            Ok((rest, Literal::Bool(true)))
        }
        Some(Token::False) => {
            let (rest, _) = consume_one(input)?;
            Ok((rest, Literal::Bool(false)))
        }
        Some(Token::Unit) => {
            let (rest, _) = consume_one(input)?;
            Ok((rest, Literal::Unit))
        }
        _ => Err(NomError(Error::tag(input))),
    }
}

// ========================= LOW LEVEL TOKEN HELPERS ===================

fn peek_token(input: Tokens) -> Option<Token> {
    input.iter_elements().next()
}

fn consume_one(input: Tokens) -> ParseResult<Token> {
    match input.iter_elements().next() {
        Some(tok) => Ok((input.take_from(1), tok)),
        None => Err(NomError(Error::eof(input))),
    }
}

fn consume_expected_delimiter(input: Tokens, expected: Delimiter) -> ParseResult<Delimiter> {
    match input.iter_elements().next() {
        Some(Token::Delimiter(d)) if d == expected => Ok((input.take_from(1), d)),
        Some(_) => Err(NomError(Error::tag(input))),
        None => Err(NomError(Error::eof(input))),
    }
}

fn consume_expected_punctuation(input: Tokens, expected: Punctuation) -> ParseResult<Punctuation> {
    match input.iter_elements().next() {
        Some(Token::Punctuation(p)) if p == expected => Ok((input.take_from(1), p)),
        Some(_) => Err(NomError(Error::tag(input))),
        None => Err(NomError(Error::eof(input))),
    }
}

fn consume_identifier(input: Tokens) -> ParseResult<Identifier> {
    match input.iter_elements().next() {
        Some(Token::Identifier(id)) => Ok((input.take_from(1), id)),
        Some(_) => Err(NomError(Error::tag(input))),
        None => Err(NomError(Error::eof(input))),
    }
}

fn expect_delimiter(del: Delimiter) -> impl FnMut(Tokens) -> ParseResult<Delimiter> {
    move |input: Tokens| {
        let (input, seen) = parse_variant!(Delimiter).parse(input)?;
        if seen == del {
            Ok((input, seen))
        } else {
            Err(NomError(Error::tag(input)))
        }
    }
}

// TODO: Not super happy about this, want to use `delimited` here
fn surrounded_by<F, O>(
    left: Delimiter,
    inside: F,
    right: Delimiter,
) -> impl Fn(Tokens) -> ParseResult<O>
where
    F: Fn(Tokens) -> ParseResult<O>,
{
    move |input| {
        let (input, _) = expect_delimiter(left)(input)?;
        let (input, out) = inside(input)?;
        let (input, _) = expect_delimiter(right)(input)?;
        Ok((input, out))
    }
}

fn parse_identifier(input: Tokens) -> ParseResult<Identifier> {
    parse_variant!(Identifier).parse(input)
}

fn parse_keyword(input: Tokens) -> ParseResult<Keyword> {
    parse_variant!(Keyword).parse(input)
}

fn parse_num_lit(tokens: Tokens) -> ParseResult<LiteralNumber> {
    let next_token = tokens.iter_elements().next();

    if let Some(num) = next_token {
        match num {
            Token::Number(lit_num) => return Ok((tokens, lit_num)),
            _ => return Err(NomError(Error::tag(tokens))),
        }
    } else {
        return Err(NomError(Error::eof(tokens)));
    }
}

fn parse_dec_num_lit(tokens: Tokens) -> ParseResult<usize> {
    let (tokens, num) = parse_num_lit(tokens)?;
    match num {
        LiteralNumber::DecInt(x) => {
            if x >= 0 {
                return Ok((tokens, x as usize));
            } else {
                return Err(NomError(Error::tag(tokens)));
            }
        }
        _ => return Err(NomError(Error::tag(tokens))),
    }
}

fn expect_keywords(expected: Vec<Keyword>) -> impl Fn(Tokens) -> ParseResult<Keyword> {
    move |input: Tokens| {
        let (input, keyword) = parse_keyword(input)?;
        if expected.contains(&keyword) {
            Ok((input, keyword))
        } else {
            Err(NomError(Error::expected_keyword(expected.clone(), keyword)))
        }
    }
}

fn parse_atom<T: TokenExtract>() -> impl Fn(Tokens) -> ParseResult<T> {
    move |input: Tokens| match input.iter_elements().next() {
        Some(token) => match T::extract(token) {
            Some(value) => Ok((input.take_from(1), value)),
            None => Err(NomError(Error::tag(input))),
        },
        None => Err(NomError(Error::eof(input))),
    }
}

fn expect_atomic_token(token: Token) -> impl FnMut(Tokens) -> ParseResult<Token> {
    move |input: Tokens| match input.iter_elements().next() {
        Some(seen) if seen == token => Ok((input.take_from(1), seen.clone())),
        Some(_) => Err(NomError(Error::tag(input))),
        None => Err(NomError(Error::eof(input))),
    }
}
