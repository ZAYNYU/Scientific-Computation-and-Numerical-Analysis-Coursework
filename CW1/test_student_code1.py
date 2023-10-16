"""
Created in June 2021

@author: pmzejh
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
    'Q2 Case I': {
        'GeneralCommands': {
            'Import': 'rootfinders as rf',
            'Input1': 'f = lambda x: x**3 + x**2 - 2*x - 2',
            'Input2': 'a = 1',
            'Input3': 'b = 2',
            'Input4': 'Nmax = 5',
            'Command1': {'cmd': 'rf.bisection(f,a,b,Nmax)', 'Output1': 'p_array'},
            'Command2': {'cmd': 'type(p_array)', 'Output1': 'p_array_type'},
            'Command3': {'cmd': 'np.shape(p_array)','Output1': 'p_array_shape'},
            },
        'Tests' : {
            'Test: p_array_type' : {
                'TestObject': 'p_array_type',
                'ObjectType': type},
            'Test: p_array_shape': {
                'TestObject': 'p_array_shape',
                'ObjectType': tuple},
            'Test: p_array' : {
                'TestObject': 'p_array',
                'ObjectType': np.ndarray}
            }
        },
    'Q2 Case II': {
        'GeneralCommands': {
            'Import': 'rootfinders as rf',
            'Input1': 'f = lambda x: x**3 + x**2 - 2*x - 2',
            'Input2': 'a = 1',
            'Input3': 'b = 3',
            'Input4': 'Nmax = 15',
            'Command1': {'cmd': 'rf.bisection(f,a,b,Nmax)', 'Output1': 'p_array'},
            },
       'Tests' : {
            'Test: p_array' : {
                'TestObject': 'p_array',
                'ObjectType': np.ndarray},
            }
        },
   'Q3 Case I': {
        'GeneralCommands': {
            'Import': 'rootfinders as rf',
            'Input1': 'f = lambda x: x**3 + x**2 - 2*x - 2',
            'Input2': 'c = 1/12',
            'Input3': 'p0 = 1',
            'Input4': 'Nmax = 10',
            'Command1': {'cmd': 'rf.fixedpoint_iteration(f,c,p0,Nmax)', 'Output1': 'p_array'},
            'Command2': {'cmd': 'type(p_array)', 'Output1': 'p_array_type'},
            'Command3': {'cmd': 'np.shape(p_array)','Output1': 'p_array_shape'},
            'Command4': {'cmd': 'rf.fixedpoint_iteration.__doc__','Output1':'doc_string'},
            },
        'Tests' : {
            'Test: p_array_type' : {
                'TestObject': 'p_array_type',
                'ObjectType': type},
            'Test: p_array_shape': {
                'TestObject': 'p_array_shape',
                'ObjectType': tuple},
            'Test: p_array' : {
                'TestObject': 'p_array',
                'ObjectType': np.ndarray},
            'Test: help(p_array)' : {
                'TestObject': 'doc_string',
                'ObjectType': str}
            }
        },
    'Q3 Case II': {
        'GeneralCommands': {
            'Import': 'rootfinders as rf',
            'Input1': 'f = lambda x: x**3 + x**2 - 2*x - 2',
            'Input2': 'c = 1/2',
            'Input3': 'p0 = 0',
            'Input4': 'Nmax = 20',
            'Command1': {'cmd': 'rf.fixedpoint_iteration(f,c,p0,Nmax)', 'Output1': 'p_array'},
            },
       'Tests' : {
            'Test: p_array' : {
                'TestObject': 'p_array',
                'ObjectType': np.ndarray},
            }
        },    
   'Q4 Case I': {
        'GeneralCommands': {
            'Import': 'rootfinders as rf',
            'Input1': 'f = lambda x: x**2 - 2',
            'Input2': 'dfdx = lambda x: 2*x',
            'Input3': 'p0 = 1',
            'Input4': 'Nmax = 8',
            'Command1': {'cmd': 'rf.newton_method(f,dfdx,p0,Nmax)', 'Output1': 'p_array'},
            'Command2': {'cmd': 'type(p_array)', 'Output1': 'p_array_type'},
            'Command3': {'cmd': 'np.shape(p_array)','Output1': 'p_array_shape'},
            },
        'Tests' : {
            'Test: p_array_type' : {
                'TestObject': 'p_array_type',
                'ObjectType': type},
            'Test: p_array_shape': {
                'TestObject': 'p_array_shape',
                'ObjectType': tuple},
            'Test: p_array' : {
                'TestObject': 'p_array',
                'ObjectType': np.ndarray},
            }
        },
    'Q4 Case II': {
        'GeneralCommands': {
            'Import': 'rootfinders as rf',
            'Input1': 'f = lambda x: x**3 + x**2 - 2*x - 2',
            'Input2': 'dfdx = lambda x: 3*x**2 + 2*x - 2',
            'Input3': 'p0 = 1',
            'Input4': 'Nmax = 8',
            'Command1': {'cmd': 'rf.newton_method(f,dfdx,p0,Nmax)', 'Output1': 'p_array'},
            },
       'Tests' : {
            'Test: p_array' : {
                'TestObject': 'p_array',
                'ObjectType': np.ndarray},
            }
        },        
    'Q5': {
        'GeneralCommands': {
            'Import': 'rootfinders as rf',
            'Input1': 'f = lambda x: x**3 + x**2 - 2*x - 2',
            'Input2': 'dfdx = lambda x: 3*x**2 + 2*x - 2',
            'Input3': 'c = 1/12',
            'Input4': 'p0 = 1',
            'Input5': 'p1 = 2',
            'Input6': 'Nmax = 20',
            'Input7': 'p_exact = np.sqrt(2)',
            'Command1': {'cmd': 'plt.figure()', 'Output1': 'fig1'},
            'Command2': {'cmd': 'rf.plot_convergence(p_exact,f,dfdx,c,p0,p1,Nmax,fig1)', 'Output1': 'out'},
            'Command3': {'cmd': 'plt.gcf()', 'Output1': 'fig'},
            },
       'Tests' : {
            'Test: fig' : {
                'TestObject': 'fig',
                'ObjectType': matplotlib.figure.Figure},
            }
       },
   'Q6 Case I': {
        'GeneralCommands': {
            'Import': 'rootfinders as rf',
            'Input1': 'f = lambda x: x**2 - 2',
            'Input2': 'p0 = 1',
            'Input3': 'p1 = 2',
            'Input4': 'Nmax = 5',
            'Command1': {'cmd': 'rf.secant_method(f,p0,p1,Nmax)', 'Output1': 'p_array'},
            'Command2': {'cmd': 'type(p_array)', 'Output1': 'p_array_type'},
            'Command3': {'cmd': 'np.shape(p_array)','Output1': 'p_array_shape'},
            },
        'Tests' : {
            'Test: p_array_type' : {
                'TestObject': 'p_array_type',
                'ObjectType': type},
            'Test: p_array_shape': {
                'TestObject': 'p_array_shape',
                'ObjectType': tuple},
            'Test: p_array' : {
                'TestObject': 'p_array',
                'ObjectType': np.ndarray},   
        }
    },
  'Q6 Case II': {
        'GeneralCommands': {
            'Import': 'rootfinders as rf',
            'Input1': 'f = lambda x: x**2 - 2',
            'Input2': 'p0 = 1',
            'Input3': 'p1 = 2',
            'Input4': 'Nmax = 12',
            'Command1': {'cmd': 'rf.secant_method(f,p0,p1,Nmax)', 'Output1': 'p_array'},
            },
        'Tests' : {
            'Test: p_array' : {
                'TestObject': 'p_array',
                'ObjectType': np.ndarray},   
        }
    }   
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

plt.show2 = plt.show
plt.show = lambda x=1: x

for case_name in case_keys:
    
    print("      Running ",case_name)
    #Run the student code and store output
    run_test_case(case_definitions_dict,case_name,student_command_window)

create_html_of_outputs(case_definitions_dict,student_command_window)
print("      Created file StudentCodeTestOutput.html")
print("          (open it in a web browser)")

plt.show = plt.show2
