try:
    from IPython.core.magic import register_line_magic
except ImportError:
    def register_line_magic(func):
        return func

@register_line_magic
def start_checks(line=''):
    """Stub start_checks magic for non-graded environments."""
    return None
