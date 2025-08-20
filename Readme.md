![alt text](image.png)



1. Allow Scripts to Run (PowerShell)

In PowerShell, we first need to allow scripts to run temporarily. This will only affect the current PowerShell session.

Run the following command in PowerShell:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass


This will allow you to activate the virtual environment.

2. Activate the tf_env Virtual Environment

Since you are already in the C:\Jame\neural_network directory, you can activate your tf_env environment by running this command:

.\tf_env\Scripts\Activate.ps1


Once you run this, the prompt should change to show that you're inside the tf_env environment, like this:

(tf_env) C:\Jame\neural_network>
