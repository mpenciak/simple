use nom::{branch::alt, combinator::opt, multi::separated_list0, sequence::preceded, Parser};

use crate::{
    ast::{Block, Expr, LValue, LetStatement, Statement},
    lexer::{Delimiter, Keyword, Punctuation, Token, Tokens},
    parser::{
        expect_atomic_token, expect_keywords,
        expr::{parse_dec_num_lit, parse_expr},
        parse_identifier, surrounded_by,
        types::parse_type_expr,
        ParseResult,
    },
};

pub fn parse_block(input: Tokens) -> ParseResult<Block> {
    separated_list0(
        expect_atomic_token(Token::Punctuation(Punctuation::Semicolon)),
        parse_statement,
    )
    .map(|statements| Block { statements })
    .parse(input)
}

pub fn parse_statement(input: Tokens) -> ParseResult<Statement> {
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
