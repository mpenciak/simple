//! Here we define the parser

use crate::{
    errors::{self, parser::Error},
    lexer::{CoreType, Delimiter, Identifier, Keyword, Punctuation, Token, Tokens},
};
use nom::{self, Err::Error as NomError, IResult, Input, Parser};

pub mod command;
pub mod expr;
pub mod statement;
pub mod types;

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

pub(crate) use parse_variant;

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
