"""
Restricted AST Code Sandbox Guard.
Parses and validates generated Python code against security AST policies
before execution to prevent dangerous operations (file system access, shell commands, network calls).
"""
import ast
import logging
from typing import Tuple, Set

logger = logging.getLogger(__name__)

# Strictly forbidden modules that generated code must never import
FORBIDDEN_MODULES: Set[str] = {
    "os", "sys", "subprocess", "shutil", "builtins", "socket", 
    "urllib", "requests", "pickle", "ctypes", "threading", "multiprocessing",
    "pty", "platform", "signal", "importlib"
}

# Strictly forbidden built-in function calls
FORBIDDEN_BUILTINS: Set[str] = {
    "eval", "exec", "__import__", "open", "compile", 
    "getattr", "setattr", "delattr", "globals", "locals"
}

class SecurityASTVisitor(ast.NodeVisitor):
    """AST Visitor that checks for forbidden nodes, imports, and calls."""
    
    def __init__(self):
        self.is_secure = True
        self.violation_reason = ""
        
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            module_name = alias.name.split(".")[0]
            if module_name in FORBIDDEN_MODULES:
                self.is_secure = False
                self.violation_reason = f"Security Violation: Import of forbidden module '{alias.name}' is strictly prohibited."
                return
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            module_name = node.module.split(".")[0]
            if module_name in FORBIDDEN_MODULES:
                self.is_secure = False
                self.violation_reason = f"Security Violation: Import from forbidden module '{node.module}' is strictly prohibited."
                return
        self.generic_visit(node)
        
    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in FORBIDDEN_BUILTINS:
                self.is_secure = False
                self.violation_reason = f"Security Violation: Calling forbidden builtin function '{func_name}()' is strictly prohibited."
                return
        self.generic_visit(node)

def validate_code_security(code: str) -> Tuple[bool, str]:
    """
    Validates generated Python code using AST static analysis.
    
    Args:
        code (str): The Python code string to inspect.
        
    Returns:
        Tuple[bool, str]: (True, "Valid") if secure, or (False, error_msg) if security violation found.
    """
    if not code or not code.strip():
        return True, "Empty code block"
        
    try:
        parsed_ast = ast.parse(code)
    except SyntaxError as se:
        logger.warning(f"Syntax error in generated code: {se}")
        return False, f"Syntax Error: {se}"
        
    visitor = SecurityASTVisitor()
    visitor.visit(parsed_ast)
    
    if not visitor.is_secure:
        logger.error(visitor.violation_reason)
        return False, visitor.violation_reason
        
    return True, "Valid and Secure"
