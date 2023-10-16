"""
Created in June 2021

@author: pmzejh / RR
"""

import numpy as np
import time
import re
import importlib
import matplotlib
import matplotlib.pyplot as plt
import errno


############################################################
case_definitions_dict = {
    'Q1 Case I': {
        'GeneralCommands': {
            'Import': 'systemsolvers as ss',
            'Input1': 'A = np.array([[1,-5,1],[10,0.0,20],[5,10,-1]])',
            'Input2': 'b = np.array([[7],[6],[4]])',
            'Input3': 'n = 3',
            'Input4': 'c = 1',
            'Command1': {'cmd': 'ss.no_pivoting(A,b,n,c)', 'Output1': 'M'},
            'Command2': {'cmd': 'type(M)', 'Output1': 'M_type'},
            'Command3': {'cmd': 'np.shape(M)','Output1': 'M_shape'},
            },
        'Tests' : {
            'Test: M_type' : {
                'TestObject': 'M_type',
                'ObjectType': type},
            'Test: M_shape': {
                'TestObject': 'M_shape',
                'ObjectType': tuple},
            'Test: M' : {
                'TestObject': 'M',
                'ObjectType': np.ndarray}
            }
        },
    'Q1 Case II': {
        'GeneralCommands': {
            'Import': 'systemsolvers as ss',
            'Input1': 'A = np.array([[1,-5,1],[10,0.0,20],[5,10,-1]])',
            'Input2': 'b = np.array([[7],[6],[4]])',
            'Input3': 'n = 3',
            'Input4': 'c = 2',
            'Command1': {'cmd': 'ss.no_pivoting(A,b,n,c)', 'Output1': 'M'},
            },
        'Tests' : {
            'Test: M' : {
                'TestObject': 'M',
                'ObjectType': np.ndarray}
            },
        },
    'Q1 Case III': {
        'GeneralCommands': {
            'Import': 'systemsolvers as ss',
            'Input1': 'A = np.array([[1,-5,1],[10,0.0,20],[5,10,-1]])',
            'Input2': 'b = np.array([[7],[6],[4]])',
            'Input3': 'n = 3',
            'Command1': {'cmd': 'ss.no_pivoting_solve(A,b,n)', 'Output1': 'x'},
            'Command2': {'cmd': 'type(x)', 'Output1': 'x_type'},
            'Command3': {'cmd': 'np.shape(x)','Output1': 'x_shape'},
            'Command4': {'cmd': 'ss.no_pivoting_solve.__doc__','Output1':'doc_string'},
            },
        'Tests' : {
            'Test: x_type' : {
                'TestObject': 'x_type',
                'ObjectType': type},
            'Test: x_shape': {
                'TestObject': 'x_shape',
                'ObjectType': tuple},
            'Test: x' : {
                'TestObject': 'x',
                'ObjectType': np.ndarray},
            'Test: help' : {
                'TestObject': 'doc_string',
                'ObjectType': str}
            }
        },
    'Q2 Case I': {
        'GeneralCommands': {
            'Import': 'systemsolvers as ss',
            'Input1': 'M = np.array([[1,-5,1,7],[10,0.0,20,6],[5,10,-1,4]])',
            'Input2': 'n = 3',
            'Input3': 'i = 0',
            'Command1': {'cmd': 'ss.find_max(M,n,i)', 'Output1': 'p'},
            'Command2': {'cmd': 'type(p)', 'Output1': 'p_type'},
            },
        'Tests' : {
            'Test: p_type' : {
                'TestObject': 'p_type',
                'ObjectType': type},
            'Test: p' : {
                'TestObject': 'p',
                'ObjectType': int}
            }
        },
    'Q2 Case II': {
        'GeneralCommands': {
            'Import': 'systemsolvers as ss',
            'Input1': 'A = np.array([[1,-5,1],[10,0.0,20],[5,10,-1]])',
            'Input2': 'b = np.array([[7],[6],[4]])',
            'Input3': 'n = 3',
            'Input4': 'c = 1',
            'Command1': {'cmd': 'ss.partial_pivoting(A,b,n,c)', 'Output1': 'M'},
            'Command2': {'cmd': 'type(M)', 'Output1': 'M_type'},
            'Command3': {'cmd': 'np.shape(M)','Output1': 'M_shape'},
            },
        'Tests' : {
            'Test: M_type' : {
                'TestObject': 'M_type',
                'ObjectType': type},
            'Test: M_shape': {
                'TestObject': 'M_shape',
                'ObjectType': tuple},
            'Test: M' : {
                'TestObject': 'M',
                'ObjectType': np.ndarray}
            },
        },
    'Q2 Case III': {
        'GeneralCommands': {
            'Import': 'systemsolvers as ss',
            'Input1': 'M = np.array([[10,0,20,6],[0,-5,-1,6.4],[0,10,-11,1]])',
            'Input2': 'n = 3',
            'Input3': 'i = 1',
            'Command1': {'cmd': 'ss.find_max(M,n,i)', 'Output1': 'p'},
            },
        'Tests' : {
            'Test: p' : {
                'TestObject': 'p',
                'ObjectType': int}
            }
        },
    'Q2 Case IV': {
        'GeneralCommands': {
            'Import': 'systemsolvers as ss',
            'Input1': 'A = np.array([[1,-5,1],[10,0.0,20],[5,10,-1]])',
            'Input2': 'b = np.array([[7],[6],[4]])',
            'Input3': 'n = 3',
            'Input4': 'c = 2',
            'Command1': {'cmd': 'ss.partial_pivoting(A,b,n,c)', 'Output1': 'M'},
            },
        'Tests' : {
            'Test: M' : {
                'TestObject': 'M',
                'ObjectType': np.ndarray}
            },
        },
    'Q2 Case V': {
        'GeneralCommands': {
            'Import': 'systemsolvers as ss',
            'Input1': 'A = np.array([[1,-5,1],[10,0.0,20],[5,10,-1]])',
            'Input2': 'b = np.array([[7],[6],[4]])',
            'Input3': 'n = 3',
            'Command1': {'cmd': 'ss.partial_pivoting_solve(A,b,n)', 'Output1': 'x'},
            },
        'Tests' : {
            'Test: x' : {
                'TestObject': 'x',
                'ObjectType': np.ndarray},
            }
        },
    'Q3 Case I': {
        'GeneralCommands': {
            'Import': 'systemsolvers as ss',
            'Input1': 'A = np.array([[1,1,0.0],[2,1,-1],[0,-1,-1]])',
            'Input3': 'n = 3',
            'Command1': {'cmd': 'ss.Doolittle(A,n)', 'Output1': 'L', 'Output2': 'U'},
            'Command2': {'cmd': 'type(L)', 'Output1': 'L_type'},
            'Command3': {'cmd': 'np.shape(L)','Output1': 'L_shape'},
            'Command4': {'cmd': 'type(U)', 'Output1': 'U_type'},
            'Command5': {'cmd': 'np.shape(U)','Output1': 'U_shape'},
            },
        'Tests' : {
            'Test: L_type' : {
                'TestObject': 'L_type',
                'ObjectType': type},
            'Test: L_shape': {
                'TestObject': 'L_shape',
                'ObjectType': tuple},
            'Test: L' : {
                'TestObject': 'L',
                'ObjectType': np.ndarray},
            'Test: U_type' : {
                'TestObject': 'U_type',
                'ObjectType': type},
            'Test: U_shape': {
                'TestObject': 'U_shape',
                'ObjectType': tuple},
            'Test: U' : {
                'TestObject': 'U',
                'ObjectType': np.ndarray}
            }
        },
    'Q4 Case I': {
        'GeneralCommands': {
            'Import': 'systemsolvers as ss',
            'Input1': 'A = np.array([[4,-1,0],[-1,8,-1],[0,-1,4]])',
            'Input2': 'b = np.array([[48],[12],[24]])',
            'Input3': 'n = 3',
            'Input4': 'x0 = np.array([[1.0],[1],[1]])',
            'Input5': 'tol = 1e-2',
            'Input6': 'maxits = 3',
            'Command1': {'cmd': 'ss.Gauss_Seidel(A,b,n,x0,tol,maxits)', 'Output1': 'x'},
            'Command2': {'cmd': 'type(x)', 'Output1': 'x_type'},
            'Command3': {'cmd': 'np.shape(x)','Output1': 'x_shape'},
            },
        'Tests' : {
            'Test: x_type' : {
                'TestObject': 'x_type',
                'ObjectType': type},
            'Test: x' : {
                'TestObject': 'x',
                'ObjectType': str}
            }
        },
    'Q4 Case II': {
        'GeneralCommands': {
            'Import': 'systemsolvers as ss',
            'Input1': 'A = np.array([[4,-1,0],[-1,8,-1],[0,-1,4]])',
            'Input2': 'b = np.array([[48],[12],[24]])',
            'Input3': 'n = 3',
            'Input4': 'x0 = np.array([[1.0],[1],[1]])',
            'Input5': 'tol = 1e-2',
            'Input6': 'maxits = 4',
            'Command1': {'cmd': 'ss.Gauss_Seidel(A,b,n,x0,tol,maxits)', 'Output1': 'x'},
            'Command2': {'cmd': 'type(x)', 'Output1': 'x_type'},
            'Command3': {'cmd': 'np.shape(x)','Output1': 'x_shape'},
            },
        'Tests' : {
            'Test: x_type' : {
                'TestObject': 'x_type',
                'ObjectType': type},
            'Test: x_shape': {
                'TestObject': 'x_shape',
                'ObjectType': tuple},
            'Test: x' : {
                'TestObject': 'x',
                'ObjectType': np.ndarray}
            }
        },
    }




############################################################
def run_test_case(dict_name,case_name,student_command_window):
    
    """
    Runs test case case_name from the dict_name
    It is assumed all files to import are in the current direcotry
    """

    student_command_window[case_name] = "<tt>"
    
    key_list = list(dict_name[case_name]["GeneralCommands"].keys())
    
    glob_dict = globals()
    loc_dict = {}


    #Import modules
    for key in key_list:
        if bool(re.search("^Import",key)):
            cmd = "import "+dict_name[case_name]["GeneralCommands"].get(key)

            student_command_window[case_name] = student_command_window[case_name]+cmd+"<br>\n"
            try:                
                exec(cmd,glob_dict,loc_dict)
            except Exception as e:
                student_command_window[case_name] = student_command_window[case_name]+f'<span style="color:red"> Error Raised: {e}</span><br>\n'
                return

    #Variable set up (based on Inputs in dictionary)
    for key in key_list:
        if bool(re.search("^Input",key)):
            cmd = dict_name[case_name]["GeneralCommands"].get(key)
            student_command_window[case_name] = student_command_window[case_name]+cmd+"<br>\n"
            try:
                exec(cmd,glob_dict,loc_dict)
            except Exception as e:
                student_command_window[case_name] = student_command_window[case_name]+f'<span style="color:red"> Error Raised: {e}</span><br>\n'

    #Initialise the output dictionary
    dict_name[case_name]["Outputs"] = {}
    
    #Run the set of commands
    
    for key in key_list:
            if bool(re.search("^Command",key)):
                #Set up the outputs
                command_key_list = list(dict_name[case_name]["GeneralCommands"][key].keys())
                Outputs = ""
                for cmd_key in command_key_list:
                    if bool(re.search("^Output",cmd_key)):
                        Outputs = Outputs + dict_name[case_name]["GeneralCommands"][key].get(cmd_key) + ", "

                if len(Outputs) >= 3:
                    Outputs = Outputs[0:len(Outputs)-2]

                cmd = Outputs + " = " + dict_name[case_name]["GeneralCommands"][key].get("cmd")

                student_command_window[case_name] = student_command_window[case_name]+cmd+"<br>\n"
                try:
                    exec(cmd,glob_dict,loc_dict)
                        
                    #Append the outputs to the Outputs section of the case dictionary
                    for cmd_key in command_key_list:
                        if bool(re.search("^Output",cmd_key)):
                            output_name = dict_name[case_name]["GeneralCommands"][key].get(cmd_key)
                            dict_name[case_name]["Outputs"][output_name] = loc_dict.get(output_name)
                except Exception as e:
                    student_command_window[case_name] = student_command_window[case_name]+f'<span style="color:red"> Error Raised: {e}</span><br>\n'

    student_command_window[case_name] = student_command_window[case_name]+'</tt>'


############################################################

def create_html_of_outputs(student_case_dict,cmd_window):
    

    with open('StudentCodeTestOutput.html','w') as file:
        file.writelines('\n'.join(["<!DOCTYPE html>","<html>"]))
        file.write('<head> \n')
        file.write('Output from Code tests')
        file.write('</head> \n')

        file.write('<body> \n')

        case_keys = student_case_dict.keys()

        for case_name in case_keys:

            file.write('<p> <b>Case: '+case_name+'</b><br></p>\n')
            
            #Output the commands run
            file.write('<p style="margin-left:30px;"> <u>Commands Run:</u> </p>\n')
            file.write('<p style="margin-left:60px;"<tt>'+cmd_window[case_name]+'</tt></p>')
            
            key_list = list(student_case_dict[case_name]["Tests"].keys())

            for key in key_list:
                if bool(re.search("^Test",key)):
                    file.write('<pre><p style="margin-left:30px;"><u>'+key+'</u><br></p></pre>\n')
                    test_object_key = student_case_dict[case_name]["Tests"][key].get("TestObject")

                    if "Outputs" in student_case_dict[case_name]:
                            
                        student_output = student_case_dict[case_name]["Outputs"].get(test_object_key)

                        file.write('<p style="margin-left:60px;"> Student Output: </p>\n')
                        

                        student_output_type = type(student_output)
                        required_output_type = student_case_dict[case_name]["Tests"][key].get("ObjectType")
                        
                        if student_output_type != required_output_type:
                            required_string = str(required_output_type).replace("<","&lt")
                            required_string = required_string.replace(">","&gt")
                            received_string = str(student_output_type).replace("<","&lt")
                            received_string = received_string.replace(">","&gt")
                            warn = "requires <tt>"+required_string+"</tt> received <tt>"+received_string+"</tt>"
                            file.write("<p style=\"margin-left:60px;\"><span style=\"color:red\">Warning</span>: Student output is of incorrect type, "+warn+"</p>\n")

                        file.write('<pre><p style="margin-left:60px;">'+ test_object_key + ' = </p></pre>')
                            
                        if isinstance(student_output,matplotlib.figure.Figure):
                            student_output.savefig(test_object_key+".pdf")
                            
                            file.write('<p style="margin-left:90px;"><iframe src=\"'+test_object_key+".pdf\" height=\"400\" width=\"600\"></iframe></p><br><br>\n")
                        else:
                            student_output = str(student_output)
                            
                            #if isinstance(student_output,str):
                            student_output = student_output.replace("<","&lt")
                            student_output = student_output.replace(">","&gt")
                            student_output = student_output.replace('\n','<br>')
                            #student_output = '<pre> '+student_output+' </pre>'
                            
                            file.write('<pre><p style="margin-left:90px;">'+ student_output +'</p></pre>')
                    else:
                        file.write('<p style="margin-left:60px;"> Student Output: None </p>\n')

                
                        
        file.write('</body> \n')
        file.write('</html> \n')


############################################################
#Test the code and output

case_keys = case_definitions_dict.keys()

student_command_window = {}

for case_name in case_keys:

    print("      Running ",case_name)
    #Run the student code and store output
    run_test_case(case_definitions_dict,case_name,student_command_window)

create_html_of_outputs(case_definitions_dict,student_command_window)
print("      Created file StudentCodeTestOutput.html")
print("          (open it in a web browser)")


