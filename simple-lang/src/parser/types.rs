use nom::{branch::alt, multi::separated_list0, Parser};

use crate::{
    ast::TypeExpr,
    lexer::{CoreType, Delimiter, Keyword, Punctuation, Token, Tokens},
    parser::{
        expect_atomic_token, expect_keywords, parse_atom, parse_identifier, parse_variant,
        surrounded_by, ParseResult,
    },
};

pub fn parse_type_expr(input: Tokens) -> ParseResult<TypeExpr> {
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
