from pathlib import Path
from enum import Enum
import os
from typing import Iterable


class TokenType(Enum):
    KEYWORD = "keyword"
    SYMBOL = "symbol"
    IDENTIFIER = "identifier"
    INT_CONST = "int_const"
    STRING_CONST = "string_const"

    @classmethod
    def get_values(cls) -> list[str]:
        return [e.value for e in cls]

    def get_xml_tag_value(self) -> str:
        match self:
            case TokenType.INT_CONST:
                return "integerConstant"
            case TokenType.STRING_CONST:
                return "stringConstant"
            case _:
                return self.value


class KeyWord(Enum):
    CLASS = "class"
    METHOD = "method"
    FUNCTION = "function"
    CONSTRUCTOR = "constructor"
    INT = "int"
    BOOLEAN = "boolean"
    CHAR = "char"
    VOID = "void"
    VAR = "var"
    STATIC = "static"
    FIELD = "field"
    LET = "let"
    DO = "do"
    IF = "if"
    ELSE = "else"
    WHILE = "while"
    RETURN = "return"
    TRUE = "true"
    FALSE = "false"
    NULL = "null"
    THIS = "this"

    @classmethod
    def get_values(cls) -> list[str]:
        return [e.value for e in cls]


class JackTokenizer:
    symbols = {
        "{",
        "}",
        "(",
        ")",
        "[",
        "]",
        ".",
        ",",
        ";",
        "+",
        "-",
        "*",
        "/",
        "&",
        "|",
        "<",
        ">",
        "=",
        "_",
        "~",
    }
    symbols_map = {"<": "&lt;", ">": "&gt;", "&": "&amp;", "“": "&quot;"}
    content: str
    tokens: list[tuple[TokenType, str | int]]
    file_name: str
    current_index: int = 0

    def __init__(self, path: str):
        with open(path, "r") as content_file:
            self.content = content_file.read()

        self.file_name = Path(path).stem
        self.tokens = []
        self.__remove_comments()
        self.__tokenize_content()

    def __remove_comments(self) -> None:
        idx = 0
        new_content: list[str] = []
        while idx < len(self.content):
            if self.content[idx] == "/" and (
                self.content[idx + 1] == "/" or self.content[idx + 1] == "*"
            ):
                idx = idx + 1
                if self.content[idx] == "/":
                    idx = idx + 1
                    while idx < len(self.content) and self.content[idx] != "\n":
                        idx = idx + 1
                    idx = idx + 1
                    new_content.append("\n")
                else:
                    idx = idx + 2
                    while idx < len(self.content):
                        if self.content[idx] == "*" and self.content[idx + 1] == "/":
                            break
                        idx = idx + 1
                    idx = idx + 2
            else:
                if idx < len(self.content):
                    new_content.append(self.content[idx])
                idx = idx + 1

        self.content = "".join(new_content).strip()

    def __parse_str(self, idx) -> tuple[str, int]:
        new_str: list[str] = []
        while idx < len(self.content) and self.content[idx] != '"':
            new_str.append(self.content[idx])
            idx = idx + 1
        parsed_str = "".join(new_str)

        return (TokenType.STRING_CONST, parsed_str), idx + 1

    def __parse_int(self, idx) -> tuple[int, int]:
        new_int: list[str] = []
        while (
            idx < len(self.content)
            and self.content[idx] >= "0"
            and self.content[idx] <= "9"
        ):
            new_int.append(self.content[idx])
            idx = idx + 1

        parsed_int = int("".join(new_int))

        return (TokenType.INT_CONST, parsed_int), idx

    def __parse_kwd_or_identifier(self, idx) -> tuple[str, int]:
        new_str: list[str] = []
        while idx < len(self.content) and (
            self.content[idx].isalnum() or self.content[idx] == "_"
        ):
            new_str.append(self.content[idx])
            idx = idx + 1

        parsed_str = "".join(new_str)
        if parsed_str in KeyWord.get_values():
            return (TokenType.KEYWORD, parsed_str), idx

        return (TokenType.IDENTIFIER, parsed_str), idx

    def __tokenize_content(self) -> None:
        idx = 0
        while idx < len(self.content):
            char = self.content[idx]

            if char in self.symbols:
                self.tokens.append((TokenType.SYMBOL, char))
                idx = idx + 1
            elif char == '"':
                parsed_str, new_idx = self.__parse_str(idx + 1)
                self.tokens.append(parsed_str)
                idx = new_idx
            elif char >= "0" and char <= "9":
                parsed_int, new_idx = self.__parse_int(idx)
                self.tokens.append(parsed_int)
                idx = new_idx
            elif char.isalnum() or char == "_":
                parsed_str, new_idx = self.__parse_kwd_or_identifier(idx)
                self.tokens.append(parsed_str)
                idx = new_idx
            else:
                idx = idx + 1

    def __prepare_path(self, path: str) -> None:
        return f"{path}/{self.file_name}T.xml"

    def write_tokens(self, path: str) -> None:
        path = self.__prepare_path(path)
        file_content = []
        file_content.append("<tokens>\n")
        for token in self.tokens:
            token_type = token[0].get_xml_tag_value()
            token_value = (
                token[1]
                if token[1] not in self.symbols_map
                else self.symbols_map[token[1]]
            )
            token_element = f"<{token_type}> {token_value} </{token_type}>\n"
            file_content.append(token_element)
        file_content.append("</tokens>\n")

        with open(path, "w") as filetowrite:
            filetowrite.write("".join(file_content))

    def has_more_tokens(self) -> bool:
        return self.current_index < len(self.tokens)

    def advance(self) -> None:
        self.current_index = self.current_index + 1

    def token_type(self) -> TokenType:
        token = self.tokens[self.current_index]
        return token[0]

    def key_word(self) -> KeyWord:
        token = self.tokens[self.current_index]
        return KeyWord(token[1])

    def symbol(self) -> str:
        token = self.tokens[self.current_index]
        return token[1]

    def identifier(self) -> str:
        token = self.tokens[self.current_index]
        return token[1]

    def int_val(self) -> int:
        token = self.tokens[self.current_index]
        return token[1]

    def string_val(self) -> int:
        token = self.tokens[self.current_index]
        return token[1]


class CompilationEngine:
    tokenizer: JackTokenizer
    content: list[str]

    def __init__(self, tokenizer: JackTokenizer):
        self.tokenizer = tokenizer
        self.content = []

    def compile_class(self) -> None:
        self.content.append("<class> ")
        self.__compile_keyword(KeyWord.CLASS)
        self.__compile_identifier()
        self.__compile_symbol("{")
        self.__compile_class_var_dec()
        self.__compile_subroutine_dec()
        self.__compile_symbol("}")
        self.content.append("</class>")

    def __compile_identifier(self) -> None:
        identifier = self.tokenizer.identifier()
        self.tokenizer.advance()
        self.content.append(f"<identifier> {identifier} </identifier>")

    def __compile_keyword(
        self, expected_kwd: Iterable[KeyWord] | KeyWord | None = None
    ) -> None:
        kwd = self.tokenizer.key_word()
        if expected_kwd:
            if (isinstance(expected_kwd, Iterable) and kwd not in expected_kwd) or (
                isinstance(expected_kwd, KeyWord) and kwd != expected_kwd
            ):
                raise Exception(f"Expected keyword: '{expected_kwd}' got {kwd}")
        self.tokenizer.advance()
        self.content.append(f"<keyword> {kwd.value} </keyword>")

    def __compile_int_const(self) -> None:
        value = self.tokenizer.int_val()
        self.tokenizer.advance()
        self.content.append(f"<integerConstant> {value} </integerConstant>")

    def __compile_string_const(self) -> None:
        value = self.tokenizer.int_val()
        self.tokenizer.advance()
        self.content.append(f"<stringConstant> {value} </stringConstant>")

    def __compile_symbol(self, expected_symbol: str = None) -> None:
        symbol = self.tokenizer.symbol()
        if expected_symbol and symbol != expected_symbol:
            raise Exception(f"Expected symbol: '{expected_symbol}' got {symbol}")
        self.tokenizer.advance()
        self.content.append(f"<symbol> {symbol} </symbol>")

    def __compile_type(self, additional_types: Iterable[KeyWord] | None = None) -> None:
        if additional_types is None:
            additional_types = ()
        token_type = self.tokenizer.token_type()
        if token_type == TokenType.KEYWORD:
            self.__compile_keyword(
                [KeyWord.INT, KeyWord.CHAR, KeyWord.BOOLEAN, *additional_types]
            )
        else:
            self.__compile_identifier()

    def __compile_parameter_list(self) -> None:
        self.content.append(f"<parameterList> ")
        while (
            self.tokenizer.token_type() == TokenType.KEYWORD
            or self.tokenizer.token_type() == TokenType.IDENTIFIER
        ):
            self.__compile_type()
            self.__compile_identifier()
            if (
                self.tokenizer.token_type() == TokenType.SYMBOL
                and self.tokenizer.symbol() == ","
            ):
                self.__compile_symbol(",")
        self.content.append(f"</parameterList> ")

    def __compile_class_var_dec(self) -> None:
        while (
            self.tokenizer.token_type() == TokenType.KEYWORD
            and self.tokenizer.key_word() in [KeyWord.STATIC, KeyWord.FIELD]
        ):
            self.content.append(f"<classVarDec> ")
            self.__compile_keyword([KeyWord.STATIC, KeyWord.FIELD])
            self.__compile_type()
            self.__compile_identifier()
            while (
                self.tokenizer.token_type() == TokenType.SYMBOL
                and self.tokenizer.symbol() == ","
            ):
                self.__compile_symbol(",")
                self.__compile_identifier()

            self.__compile_symbol(";")
            self.content.append(f"</classVarDec> ")

    def __compile_subroutine_dec(self) -> None:
        while (
            self.tokenizer.token_type() == TokenType.KEYWORD
            and self.tokenizer.key_word()
            in [KeyWord.CONSTRUCTOR, KeyWord.FUNCTION, KeyWord.METHOD]
        ):
            self.content.append(f"<subroutineDec> ")
            self.__compile_keyword(
                [KeyWord.CONSTRUCTOR, KeyWord.FUNCTION, KeyWord.METHOD]
            )
            self.__compile_type([KeyWord.VOID])  # void | type
            self.__compile_identifier()
            self.__compile_symbol("(")
            self.__compile_parameter_list()
            self.__compile_symbol(")")
            self.__compile_subroutine_body()
            self.content.append(f"</subroutineDec> ")

    def __compile_subroutine_body(self) -> None:
        self.content.append(f"<subroutineBody> ")
        self.__compile_symbol("{")
        self.__compile_var_dec()
        self.__compile_statements()
        self.__compile_symbol("}")
        self.content.append(f"</subroutineBody> ")

    def __compile_var_dec(self) -> None:
        while (
            self.tokenizer.token_type() == TokenType.KEYWORD
            and self.tokenizer.key_word() == KeyWord.VAR
        ):
            self.content.append(f"<varDec> ")
            self.__compile_keyword(KeyWord.VAR)
            self.__compile_type()
            self.__compile_identifier()
            while (
                self.tokenizer.token_type() == TokenType.SYMBOL
                and self.tokenizer.symbol() == ","
            ):
                self.__compile_symbol(",")
                self.__compile_identifier()
            self.__compile_symbol(";")
            self.content.append(f"</varDec> ")

    def __compile_statements(self) -> None:
        self.content.append(f"<statements> ")
        while (
            self.tokenizer.token_type() == TokenType.KEYWORD
            and self.tokenizer.key_word()
            in [KeyWord.LET, KeyWord.IF, KeyWord.WHILE, KeyWord.DO, KeyWord.RETURN]
        ):
            self.__compile_statement()
        self.content.append(f"</statements> ")

    def __compile_statement(self) -> None:
        match self.tokenizer.key_word():
            case KeyWord.LET:
                self.__compile_let_statement()
            case KeyWord.IF:
                self.__compile_if_statement()
            case KeyWord.WHILE:
                self.__compile_while_statement()
            case KeyWord.DO:
                self.__compile_do_statement()
            case KeyWord.RETURN:
                self.__compile_return_statement()
            case _:
                raise Exception("No matching keyword")

    def __compile_let_statement(self) -> None:
        self.content.append(f"<letStatement> ")
        self.__compile_keyword(KeyWord.LET)
        self.__compile_identifier()
        if (
            self.tokenizer.token_type() == TokenType.SYMBOL
            and self.tokenizer.symbol() == "["
        ):
            self.__compile_symbol("[")
            self.__compile_expression()
            self.__compile_symbol("]")
        self.__compile_symbol("=")
        self.__compile_expression()
        self.__compile_symbol(";")
        self.content.append(f"</letStatement> ")

    def __compile_if_statement(self) -> None:
        self.content.append(f"<ifStatement> ")
        self.__compile_keyword(KeyWord.IF)
        self.__compile_symbol("(")
        self.__compile_expression()
        self.__compile_symbol(")")
        self.__compile_symbol("{")
        self.__compile_statements()
        self.__compile_symbol("}")
        if (
            self.tokenizer.token_type() == TokenType.KEYWORD
            and self.tokenizer.key_word() == KeyWord.ELSE
        ):
            self.__compile_keyword(KeyWord.ELSE)
            self.__compile_symbol("{")
            self.__compile_statements()
            self.__compile_symbol("}")
        self.content.append(f"</ifStatement> ")

    def __compile_while_statement(self) -> None:
        self.content.append(f"<whileStatement> ")
        self.__compile_keyword(KeyWord.WHILE)
        self.__compile_symbol("(")
        self.__compile_expression()
        self.__compile_symbol(")")
        self.__compile_symbol("{")
        self.__compile_statements()
        self.__compile_symbol("}")
        self.content.append(f"</whileStatement> ")

    def __compile_do_statement(self) -> None:
        self.content.append(f"<doStatement> ")
        self.__compile_keyword(KeyWord.DO)
        self.__compile_subroutine_call()
        self.__compile_symbol(";")
        self.content.append(f"</doStatement> ")

    def __compile_subroutine_call(self, identifier_compiled: bool = False) -> None:
        if not identifier_compiled:
            self.__compile_identifier()
        if (
            self.tokenizer.token_type() == TokenType.SYMBOL
            and self.tokenizer.symbol() == "."
        ):
            self.__compile_symbol(".")
            self.__compile_identifier()
        self.__compile_symbol("(")
        self.__compile_expression_list()
        self.__compile_symbol(")")

    def __compile_expression_list(self) -> None:
        self.content.append(f"<expressionList> ")
        if (
            self.tokenizer.token_type() == TokenType.SYMBOL
            and self.tokenizer.symbol() == ")"
        ):
            self.content.append(f"</expressionList> ")
            return
        self.__compile_expression()
        while (
            self.tokenizer.token_type() == TokenType.SYMBOL
            and self.tokenizer.symbol() == ","
        ):
            self.__compile_symbol(",")
            self.__compile_expression()
        self.content.append(f"</expressionList> ")

    def __compile_return_statement(self) -> None:
        self.content.append(f"<returnStatement> ")
        self.__compile_keyword(KeyWord.RETURN)
        if (
            self.tokenizer.token_type() == TokenType.SYMBOL
            and self.tokenizer.symbol() == ";"
        ):
            self.__compile_symbol(";")
            self.content.append(f"</returnStatement> ")
            return None

        self.__compile_expression()
        self.__compile_symbol(";")
        self.content.append(f"</returnStatement> ")

    def __compile_term(self) -> None:
        self.content.append(f"<term> ")
        toke_type = self.tokenizer.token_type()
        if toke_type == TokenType.INT_CONST:
            self.__compile_int_const()
        elif toke_type == TokenType.STRING_CONST:
            self.__compile_string_const()
        elif (
            toke_type == TokenType.KEYWORD
        ):  # keyword constants true, false, null, this
            self.__compile_keyword(
                [KeyWord.TRUE, KeyWord.FALSE, KeyWord.NULL, KeyWord.THIS]
            )
        elif toke_type == TokenType.IDENTIFIER:
            self.__compile_identifier()

            if self.tokenizer.token_type() == TokenType.SYMBOL:
                if self.tokenizer.symbol() == "[":
                    self.__compile_symbol("[")
                    self.__compile_expression()
                    self.__compile_symbol("]")
                elif self.tokenizer.symbol() in ["(", "."]:
                    self.__compile_subroutine_call(identifier_compiled=True)
        elif toke_type == TokenType.SYMBOL and self.tokenizer.symbol() in ["-", "~"]:
            self.__compile_symbol()
            self.__compile_term()
        elif toke_type == TokenType.SYMBOL and self.tokenizer.symbol() == "(":
            self.__compile_symbol("(")
            self.__compile_expression()
            self.__compile_symbol(")")

        self.content.append(f"</term> ")

    def __compile_expression(self) -> None:
        self.content.append(f"<expression> ")
        self.__compile_term()
        while (
            self.tokenizer.token_type() == TokenType.SYMBOL
            and self.tokenizer.symbol() in ["+", "-", "*", "/", "&", "|", "<", ">", "="]
        ):
            self.__compile_symbol()
            self.__compile_term()
        self.content.append(f"</expression> ")

    def __prepare_path(self, path: str) -> None:
        return f"{path}/{self.tokenizer.file_name}.xml"

    def write_to_file(self, path: str) -> None:
        path = self.__prepare_path(path)
        file_content = "\n".join(self.content) + "\n"
        with open(path, "w") as filetowrite:
            filetowrite.write("".join(file_content))


class JackAnalyzer:
    def compile_source(self, path: str) -> None:
        print(path, os.path.isdir(path))
        if not os.path.exists(path) or (
            not os.path.isdir(path) and not path.endswith(".jack")
        ):
            raise Exception(f"Unknow file type: {path}")

        if os.path.isdir(path):
            analyzer_dir = f"{path}/analyzer-out"
            Path(analyzer_dir).mkdir(parents=True, exist_ok=True)

            for filename in os.scandir(path):
                if filename.is_file() and filename.path.endswith(".jack"):
                    self.__write_analyzer_output(filename.path)
        else:
            self.__write_analyzer_output(path)

    def __write_analyzer_output(self, file_path: str) -> None:
        dir_path = os.path.dirname(file_path)
        analyzer_dir = f"{dir_path}/analyzer-out"
        Path(analyzer_dir).mkdir(parents=True, exist_ok=True)

        tokenizer = JackTokenizer(file_path)
        compilation_engine = CompilationEngine(tokenizer=tokenizer)
        tokenizer.write_tokens(analyzer_dir)
        compilation_engine.compile_class()
        compilation_engine.write_to_file(analyzer_dir)


jack_analyzer = JackAnalyzer()
jack_analyzer.compile_source("./test-programs/ExpressionLessSquare")
