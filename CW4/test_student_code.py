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
import sys



############################################################
case_definitions_dict = {
    ### Q1 ###
    'Q1 Case I': {
        'GeneralCommands': {
            'Import': 'numerical_differentiation as nd',
            'Input1': 'f = lambda x: np.sin(10*x)',
            'Input2': 'h = 0.1',
            'Input3': 'x0 = 0.5',
            'Command1': {'cmd': 'nd.richardson(f,x0,h,2)', 'Output1': 'deriv_approx1'},
            'Command2': {'cmd': 'nd.richardson(f,x0,h,3)', 'Output1': 'deriv_approx2'},
          },
        'Tests' : {
            'Test: deriv_approx1' : {
                'TestObject': 'deriv_approx1',
                'ObjectType': (np.float64,float)
                },
            'Test: deriv_approx2' : {
                'TestObject': 'deriv_approx2',
                'ObjectType': (np.float64,float),
                }
        },
    },
    ### Q2 ###
    'Q2 Case I': {
        'GeneralCommands': {
                'Import': 'numerical_differentiation as nd',
                'Input1': 'f = lambda x: np.sin(10*x)',
                'Input2': 'f_deriv = lambda x: 10.0*np.cos(10*x)',
                'Input3': 'n = 20',
                'Input4': 'h_vals = np.logspace(-5,1,20)',
                'Input5': 'k_max = 4',
                'Input6': 'x0 = 0',
                'Command1': {'cmd': 'nd.richardson_errors(f,f_deriv,x0,n,h_vals,k_max)', 'Output1': 'error_matrix', 'Output2': 'fig1'},
                },
        'Tests' : {
            'Test: error_matrix' : {
                'TestObject': 'error_matrix',
                'ObjectType': (np.ndarray)
                },
            'Test: plot' : {
                'TestObject': 'fig1',
                'ObjectType': (matplotlib.figure.Figure),
                }      
            }   
    },
    'Q2 Case II': {
        'GeneralCommands': {
                'Import': 'numerical_differentiation as nd',
                'Input1': 'f = lambda x: (x<=0)*-x**2',
                'Input2': 'f_deriv = lambda x: (x<=0)*-2*x',
                'Input3': 'n = 20',
                'Input4': 'h_vals = np.logspace(-5,1,20)',
                'Input5': 'k_max = 4',
                'Input6': 'x0 = 0',
                'Command1': {'cmd': 'nd.richardson_errors(f,f_deriv,x0,n,h_vals,k_max)', 'Output1': 'error_matrix', 'Output2': 'fig2'},
                },
        'Tests' : {
            'Test: error_matrix' : {
                'TestObject': 'error_matrix',
                'ObjectType': (np.ndarray)
                },
            'Test: plot' : {
                'TestObject': 'fig2',
                'ObjectType': (matplotlib.figure.Figure),
                }      
            }   
    },
    ### Q3 ###
    'Q3 Case I': {
        'GeneralCommands': {
                'Import': 'time_steppers as ts',
                'Input1': 'a = 0; b = 2',
                'Input2': 'N = 6',
                'Input3': 'theta = 0.75',
                'Input4': 'f = lambda t,y: y-t**2+1',
                'Input5': 'df = lambda t,y: np.ones(np.shape(y))',
                'Input6': 'y0 = 0.5',
                'Command1': {'cmd': 'ts.theta_ode_solver(a,b,f,df,N,y0,theta)', 'Output1': 't', 'Output2': 'y'},
                },
        'Tests' : {
            'Test: t' : {
                'TestObject': 't',
                'ObjectType': (np.ndarray)
                },
            'Test: y' : {
                'TestObject': 'y',
                'ObjectType': (np.ndarray),
                }      
            }
    },
    ### Q4 ###
    'Q4 Case I': {
        'GeneralCommands': {
                'Import': 'time_steppers as ts',
                'Input1': 'a = 0; b = 1',
                'Input2': 'N = 4',
                'Input3': 'm = 3',
                'Input4': 'f = lambda t,y: np.array([y[0]-2*y[1]+t*y[2],-y[0]+y[1]+y[2]-t**2,t*y[1]-y[1]+t**3])',
                'Input5': 'y0 = np.array([1,0,1])',
                'Input6': 'method = 1',
                'Command1': {'cmd': 'ts.runge_kutta(a,b,f,N,y0,m,method)', 'Output1': 't', 'Output2': 'y'},
                },
        'Tests' : {
            'Test: t' : {
                'TestObject': 't',
                'ObjectType': (np.ndarray)
                },
            'Test: y' : {
                'TestObject': 'y',
                'ObjectType': (np.ndarray),
                }      
          },
    },
    'Q4 Case II': {
        'GeneralCommands': {
                'Import': 'time_steppers as ts',
                'Input1': 'a = 0; b = 1',
                'Input2': 'N = 4',
                'Input3': 'm = 3',
                'Input4': 'f = lambda t,y: np.array([y[0]-2*y[1]+t*y[2],-y[0]+y[1]+y[2]-t**2,t*y[1]-y[1]+t**3])',
                'Input5': 'y0 = np.array([1,0,1])',
                'Input6': 'method = 2',
                'Command1': {'cmd': 'ts.runge_kutta(a,b,f,N,y0,m,method)', 'Output1': 't', 'Output2': 'y'},
                },
        'Tests' : {
            'Test: t' : {
                'TestObject': 't',
                'ObjectType': (np.ndarray)
                },
            'Test: y' : {
                'TestObject': 'y',
                'ObjectType': (np.ndarray),
                }      
          },
    },     
    'Q4 Case III': {
        'GeneralCommands': {
                'Import': 'time_steppers as ts',
                'Input1': 'a = 0; b = 1',
                'Input2': 'N = 4',
                'Input3': 'm = 3',
                'Input4': 'f = lambda t,y: np.array([y[0]-2*y[1]+t*y[2],-y[0]+y[1]+y[2]-t**2,t*y[1]-y[1]+t**3])',
                'Input5': 'y0 = np.array([1,0,1])',
                'Input6': 'method = 3',
                'Command1': {'cmd': 'ts.runge_kutta(a,b,f,N,y0,m,method)', 'Output1': 't', 'Output2': 'y'},
                },
        'Tests' : {
            'Test: t' : {
                'TestObject': 't',
                'ObjectType': (np.ndarray)
                },
            'Test: plot' : {
                'TestObject': 'y',
                'ObjectType': (np.ndarray),
                }      
        }     
    }
}

############################################################
def run_test_case(dict_name,case_name,student_command_window):
    
    """
    Runs test case case_name from the dict_name
    It is assumed all files to import are in the current directory
    """

    # Change plt.show() - to avoid pauses when running the code
    plt.show2 = plt.show
    plt.show = lambda x=1: None
    
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

    #Clean up all newly added modules
    for key in key_list:
        if bool(re.search("^Import",key)):
            if str.split(dict_name[case_name]["GeneralCommands"].get(key))[0] in sys.modules.keys(): 
                del sys.modules[str.split(dict_name[case_name]["GeneralCommands"].get(key))[0]] 
    
    # Change back plt.show()
    plt.show = plt.show2

    # Close all open figures
    plt.close('all')

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
                        
                        if not isinstance(student_output,required_output_type):
                            required_string = str(required_output_type).replace("<","&lt")
                            required_string = required_string.replace(">","&gt")
                            received_string = str(student_output_type).replace("<","&lt")
                            received_string = received_string.replace(">","&gt")
                            warn = "requires (one of) <tt>"+required_string+"</tt> received <tt>"+received_string+"</tt>"
                            file.write("<p style=\"margin-left:60px;\"><span style=\"color:red\">Warning</span>: Student output is of incorrect type, "+warn+"</p>\n")

                        file.write('<pre><p style="margin-left:60px;">'+ test_object_key + ' = </p></pre>')
                            
                        if isinstance(student_output,matplotlib.figure.Figure):
                            student_output.savefig(test_object_key+".png",bbox_inches = "tight")
                                
                            file.write('<p style="margin-left:90px;"><img src="'+test_object_key+".png\" height=\"400\" width=\"600\"></p><br><br>\n")
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


