use crate::{
    ast::{self, Block, Comparison, Expr, Literal, Parameter},
    errors::parser::Error,
    lexer::{
        self, ComparisonOperator, Delimiter, Keyword, LiteralNumber, Punctuation, Token, Tokens,
    },
    parser::{
        consume_expected_delimiter, consume_expected_punctuation, consume_identifier, consume_one,
        expect_keywords, peek_token, statement::parse_statement, types::parse_type_expr,
        ParseResult,
    },
};

use nom::{Err::Error as NomError, Input};

// Entry point for expression parsing with precedence
pub fn parse_expr(input: Tokens) -> ParseResult<Expr> {
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
                lhs = Expr::BinaryOp(ast::BinaryOperator::Or, Box::new(lhs), Box::new(rhs));
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
                lhs = Expr::BinaryOp(ast::BinaryOperator::And, Box::new(lhs), Box::new(rhs));
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
            ComparisonOperator::Eq => Comparison::Eq,
            ComparisonOperator::Ne => Comparison::Ne,
            ComparisonOperator::Lt => Comparison::Lt,
            ComparisonOperator::Le => Comparison::Le,
            ComparisonOperator::Gt => Comparison::Gt,
            ComparisonOperator::Ge => Comparison::Ge,
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
            Some(Token::BinaryOp(_op @ lexer::BinaryOperator::Add))
            | Some(Token::BinaryOp(_op @ lexer::BinaryOperator::Sub))
            | Some(Token::BinaryOp(_op @ lexer::BinaryOperator::Append)) => {
                let (after_op, token) = consume_one(rest)?;
                let (after_rhs, rhs) = parse_mul_expr(after_op)?;
                let op = match token {
                    Token::BinaryOp(lexer::BinaryOperator::Add) => ast::BinaryOperator::Add,
                    Token::BinaryOp(lexer::BinaryOperator::Sub) => ast::BinaryOperator::Sub,
                    Token::BinaryOp(lexer::BinaryOperator::Append) => ast::BinaryOperator::Append,
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
            Some(Token::BinaryOp(lexer::BinaryOperator::Mul))
            | Some(Token::BinaryOp(lexer::BinaryOperator::Div)) => {
                let (after_op, token) = consume_one(rest)?;
                let (after_rhs, rhs) = parse_unary_expr(after_op)?;
                let op = match token {
                    Token::BinaryOp(lexer::BinaryOperator::Mul) => ast::BinaryOperator::Mul,
                    Token::BinaryOp(lexer::BinaryOperator::Div) => ast::BinaryOperator::Div,
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
        Some(Token::UnaryOp(_op)) => {
            let (rest, token) = consume_one(input)?;
            let (rest, expr) = parse_unary_expr(rest)?; // allow chaining
            let uop = match token {
                Token::UnaryOp(lexer::UnaryOperator::Negative) => ast::UnaryOperator::Neg,
                Token::UnaryOp(lexer::UnaryOperator::Not) => ast::UnaryOperator::Not,
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
                if let Some(Token::Identifier(_name)) = peek_token(after_dot.clone()) {
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
            Expr::Lambda(ast::LambdaExpr {
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
        Expr::If(ast::IfExpr {
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
                Expr::Loop(ast::LoopExpr::While {
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
                Expr::Loop(ast::LoopExpr::For {
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

pub fn parse_dec_num_lit(tokens: Tokens) -> ParseResult<usize> {
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
