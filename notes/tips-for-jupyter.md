# Jupyter setup tips
- Connection refused
    - Issue: After installed jupyter, run `jupyter notebook` to start the jupyter service, it fails with connection refused error.
    - Reasons: 
        - Firewall. You need to open the port to allow the access in the windows/linux. In windows, go to the *Windows Defender Firewall* to open the port to allow access. In Linux, use the command to open the port: `sudo ufw allow 8888` (Note: replace the port to that one you need)
        - The jupyter default setting not update yet. By default, the jupyter notebook only accepts the traffic from localhost. You might need to change to config. Please check this great [instruction](https://stackoverflow.com/questions/42848130/why-i-cant-access-remote-jupyter-notebook-server). Following are the sum up steps:
          1. Create the jupyter config file: `jupyter notebook --generate-config`
          2. Edit the config file (*/home/<current user name>/.jupyter/jupyter_notebook_config.py*):           
            `c.NotebookApp.allow_origin = '*'` #allow all origins
            `c.NotebookApp.ip = '0.0.0.0'` # listen on all IPs
          3. Set the password: `jupyter notebook password` # possible [issue](https://github.com/jupyter/notebook/issues/1700)
          4. (Optional) You might want to disable open brower: `c.NotebookApp.open_browser = False`


# Jupyter efficency tips

- `%load_ext autoreload %autoreload 2`, this setting is to enable the module auto reload. If you change the source code for the module, the latest version will be automatically reloaded into the jupyter notebook environment.
- `%matplotlib inline`, enable to display the plot chart within the jupyter cell.

## Inspection related
- **?** in front of a method, view the method's docsting. Example: `?pandas.read_csv`
- **??** in front of a method, view the source code of the method. Example: `??pandas.read_csv`
- **!** execute the (shell) command. Example: `!pip install panda`

## Interop with shell
There are several ways to pass the variable value from the jupyter to the shell.
- **$** in front of the variable in shell command. Example: `!echo $a`. Another similar thing is: `!echo {a}`, please check as following.
- **{variable}**, it could use this way to pass the value to the shell command. Example:`a = '.'` `!ls {a}`, then it will list all the files in the current folder
- **%**, get the environment variable value. Example: `%pwd`

## Coding Intelligence
- **tab** for auto-completion in several contexts:
  - Varaible auto-completion. For example: `pa` and tab, then it will show list of objects/types start with pa. Btw, you could install the extension 'Hinterland' in jupyter to help you auto complete 
  - Object method/property auto-completion. For example: `pandas.`, and then tab
- **shift + tab** to show the paramsters for the function. For example: `pandas.read_csv()`, put the cursor between the parenthesis and tab
- **shift + double tab** will show parameters in detail and the examples.

## Magic functions
Magic functions are the shortcuts which are very helpful to extend the capability of jupyter. There are several frequent used ones:
- Category 1: variables check
  - **%who**, list all the varaibles' names in the current notebook context
  - **%whos**, list all the varaibles and show their types and values in current notebook context
- Category 2: file operations
  - **%pycat {file name}**, open the file content in the pager
  - **%load {file name}**, load the file content into the cell
  - **%run {script name}**, run the script. Like: test.py, test.ipynb
  - **%notebook {file name}**, export the current ipython history to a notebook file
- Category 3: envirobment operations 
  - **%env**, list all the environment variables
  - **%env {variable name}**, display the specific environment variable
  - **%env {variable name} {variable value}**, set the value for the environment variable
- Category 4: run shell commands
  - **%system {shell command}**, use the shell command. Like: `%system time` shows the current time  
- Category 5: execute langauges
  - **%%HTML**, execute the html code
  - **%%perl**, execute the perl code
  - **%%javascript** or **%%js**, execute the js code
  - **%%python3**, execute the python3 code
  - **%%ruby**, execute the ruby code
- Others
  - **%history**, print input history
  - **%lsmagic**, list all the available magic functions
  - **%magic**, print all the info about magic function system

## Performance Profiling
- **%time {command}**, traking the execution time for the command/code/ which in the same line (one command)
- **%%time**, tracking the execution time for the whole blocks after it
- **%timeit {command}**, run the command multiple times and calculate the mean and std 
- **%prun {command}**, display the exeuction time detail for the whole call stack

Notes about Wall time and CPU time:
- Wall time: the amount of actual time it takes, including CPU/IO/Networking etc.
- CPU time: the total CPU time it takes
- User time: the CPU time spent on user's code (exclude kernel code)

So, if it is running on multiple cores, the CPU time would higher than the wall time.

## Cell operations
- **esc** switch from editing to command mode
- **shift + M** in the command mode will merge the next cell into the current cell
- **shift + ctrl + -** will split the cell based on the current cursor
- **shift + enter**, it will execute the cell. Please note that, it will move to the next cell. **ctrl+enter** will execute the cell and stay there. **alt + enter** will create a new cell.
- **ctrl + [** will dedent
- **ctrl + ]** will indent

## Practice suggestion
- Self-containess: Always try to consider create the function for the code in the cell. And might parameterized the function to reuse them
  - Use the function to avoid creating a bunch of variables for different scenarios/context, and use the paramters to distinguish them. Example: `exp_sample_1k = ... ; exp_sample_2k = ...;`, you might just need to make it `exp_sample=...;` in a function and call for different experiments.
- Management: Create an import .py file if the there are many imports, and then you could simply `from imports import *`. That idea is similar like the C++ programming.
  - Or you could have all the imports done in the first cell. What is more, it would be better if all gloabl shared variables defined in the first cell from management point of view.
- Agile: start with dirty and keep the draft, then base on that keep improving/rewrite.
- Dependency management: use the *requirements.txt* to seal all the dependencies. Example: `$pip freeze > requirements.txt`

## Reference
- https://hackernoon.com/10-tips-on-using-jupyter-notebook-abc0ba7028a4
- https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/
- https://forums.fast.ai/t/jupyter-notebook-enhancements-tips-and-tricks/17064/21, a bunch of useful notes there.
- https://towardsdatascience.com/optimizing-jupyter-notebook-tips-tricks-and-nbextensions-26d75d502663
- https://github.com/NirantK/best-of-jupyter#getting-started-right, this notebook also has some more tips about the debugging.