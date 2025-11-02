use nom::{
    branch::alt,
    combinator::all_consuming,
    multi::{many0, separated_list0},
    sequence::preceded,
    Parser,
};

use crate::{
    ast::{
        Definition, FieldDef, FunctionDef, GlobalDef, ImplBlock, ImportStatement, Parameter,
        StructDef, TopLevel, TypeAlias,
    },
    lexer::{Delimiter, Keyword, Punctuation, Token, Tokens},
    parser::{
        expect_atomic_token, expect_keywords, expr::parse_expr, parse_identifier,
        statement::parse_block, surrounded_by, types::parse_type_expr, ParseResult,
    },
};

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

fn parse_import(input: Tokens) -> ParseResult<ImportStatement> {
    let (input, filename) =
        preceded(expect_keywords(vec![Keyword::Import]), parse_identifier).parse(input)?;
    Ok((input, ImportStatement { file: filename }))
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
