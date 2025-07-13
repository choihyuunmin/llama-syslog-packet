import re
import sys
import io
import traceback
from typing import Dict, Any, Optional, Tuple
import logging
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logger = logging.getLogger(__name__)

class CodeExecutor:
    def __init__(self):
        self.global_vars = {}
        self.local_vars = {}
        
    def extract_code_blocks(self, text: str) -> list:
        code_pattern = r'```(?:python)?\n(.*?)```'
        matches = re.findall(code_pattern, text, re.DOTALL)
        return [block.strip() for block in matches if block.strip()]
    
    def execute_code(self, code: str, context_data: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            if context_data:
                self.global_vars.update(context_data)
            
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute code
                exec_result = exec(code, self.global_vars, self.local_vars)
                
                # Get any return value from last expression
                if code.strip().endswith('\n') or not code.strip():
                    return_value = None
                else:
                    # Try to evaluate the last line as an expression
                    try:
                        last_line = code.strip().split('\n')[-1]
                        if not last_line.startswith(('#', 'import', 'from', 'def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except:', 'finally:', 'with ', 'async ')):
                            return_value = eval(last_line, self.global_vars, self.local_vars)
                        else:
                            return_value = None
                    except:
                        return_value = None
            
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            
            # Check for matplotlib figures
            figures = []
            if 'plt' in self.global_vars:
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    if fig:
                        # Save figure to bytes
                        import io
                        img_buffer = io.BytesIO()
                        fig.savefig(img_buffer, format='png', bbox_inches='tight')
                        img_buffer.seek(0)
                        figures.append({
                            'figure_number': fig_num,
                            'image_data': img_buffer.getvalue()
                        })
            
            return {
                'success': True,
                'stdout': stdout_output,
                'stderr': stderr_output,
                'return_value': return_value,
                'figures': figures,
                'error': None
            }
            
        except Exception as e:
            error_info = {
                'type': type(e).__name__,
                'message': str(e),
                'traceback': traceback.format_exc()
            }
            
            return {
                'success': False,
                'stdout': '',
                'stderr': '',
                'return_value': None,
                'figures': [],
                'error': error_info
            }
    
    def execute_from_response(self, response: str, context_data: Optional[Dict] = None) -> Dict[str, Any]:
        code_blocks = self.extract_code_blocks(response)
        
        if not code_blocks:
            return {
                'has_code': False,
                'message': 'No code blocks found in response'
            }
        
        results = []
        for i, code in enumerate(code_blocks):
            result = self.execute_code(code, context_data)
            result['block_index'] = i
            result['code'] = code
            results.append(result)
        
        return {
            'has_code': True,
            'results': results
        }
    
    def get_available_variables(self) -> Dict[str, Any]:
        return {
            'global_vars': list(self.global_vars.keys()),
            'local_vars': list(self.local_vars.keys())
        }
    
    def reset_environment(self):
        self.global_vars = {}
        self.local_vars = {}
        
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        self.global_vars.update({
            'pd': pd,
            'plt': plt,
            'sns': sns,
            'np': np
        }) 