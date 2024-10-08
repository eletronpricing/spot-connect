"""
Author: Carlos Valcarcel <carlos.d.valcarcel.w@gmail.com>

This file is part of spot-connect

Toolbox for launching an AWS spot instance - instance_methods.py: 

The instance_methods sub-module contains functions that interact with existing 
instances, like running a script/command, uploading a file or even terminating
the instance. 

MIT License 2020
"""

import sys
import os
import boto3
from spotconnect import ec2_methods, sutils, interactive


def run_script(instance, user_name, script, cmd=True, port=22, kp_dir=None, return_output=False):
    '''
    Run a script on the the given instance 
    __________
    parameters
    - instance : dict. Response dictionary from ec2 instance describe_instances method 
    - user_name : string. SSH username for accessing instance, default usernames for AWS images can be found at https://alestic.com/2014/01/ec2-ssh-username/
    - script : string. ".sh" file or linux/unix command (or other os resource) to execute on the instance command line 
    - cmd : if True, script string is treated as an individual argument 
    - port : port to use to connect to the instance 
    '''

    if kp_dir is None:
        kp_dir = sutils.get_default_kp_dir()

    if cmd:
        commands = script
    else:
        commands = open(script, 'r').read().replace('\r', '')

    client = ec2_methods.connect_to_instance(
        instance['PublicDnsName'], kp_dir + '/' + instance['KeyName'], username=user_name, port=port)

    session = client.get_transport().open_session()
    # Combine the error message and output message channels
    session.set_combine_stderr(True)

    # Execute a command or .sh script (unix or linux console)
    session.exec_command(commands)

    try:

        if return_output:
            output = ''

            # Collect the output
            stdout = session.makefile()

            for line in stdout:
                if return_output:
                    output += line.rstrip() + '\n'
                else:
                    # Show the output
                    print(line.rstrip(), flush=True)

    except (KeyboardInterrupt, SystemExit):
        # Keyboard interrupt
        print(sys.stderr, 'Ctrl-C, stopping', flush=True)

    # Close the connection
    client.close()

    if return_output:
        return True, output
    else:
        return True


def active_shell(instance, user_name, port=22, kp_dir=None):
    '''
    Leave a shell active
    __________
    parameters 
    - instance : dict. Response dictionary from ec2 instance describe_instances method 
    - user_name : string. SSH username for accessing instance, default usernames for AWS images can be found at https://alestic.com/2014/01/ec2-ssh-username/
    - port : port to use to connect to the instance 
    '''

    if kp_dir is None:
        kp_dir = sutils.get_default_kp_dir()

    client = ec2_methods.connect_to_instance(
        instance['PublicDnsName'], kp_dir + '/' + instance['KeyName'], username=user_name, port=port)

    console = client.invoke_shell()
    console.keep_this = client

    session = console.get_transport().open_session()
    session.get_pty()
    session.invoke_shell()

    try:
        interactive.interactive_shell(session)

    except:
        print('Logged out of interactive session.')

    session.close()
    return True


def upload_to_ec2(instance, user_name, files, remote_dir='.', kp_dir='', verbose=False):
    '''
    Upload files directly to an EC2 instance. Speed depends on internet connection and not instance type. 
    __________
    parameters 
    - instance : dict. Response dictionary from ec2 instance describe_instances method 
    - user_name : string. SSH username for accessing instance, default usernames for AWS images can be found at https://alestic.com/2014/01/ec2-ssh-username/
    - files : string or list of strings. single file, list of files or directory to upload. If it is a directory end in "/" 
    - remote_dir : '.'  string.The directory on the instance where the files will be uploaded to 
    '''

    if kp_dir == '':
        kp_dir = sutils.get_default_kp_dir()

    client = ec2_methods.connect_to_instance(
        instance['PublicDnsName'], os.path.join(kp_dir, instance['KeyName']), username=user_name, port=22)
    if verbose:
        print('Connected. Uploading files...')
    stfp = client.open_sftp()

    try:
        for f in files:
            if verbose:
                print('Uploading %s' % str(os.path.basename(f)))
            stfp.put(f, os.path.join(remote_dir, os.path.basename(f)),
                     callback=sutils.printTotals, confirm=True)

    except Exception as e:
        raise e

    if verbose:
        print('Uploaded to %s' % remote_dir)
    return True


def download_from_ec2(instance, username, get, put='.', kp_dir=None):
    '''
    Download files directly from an EC2 instance. Speed depends on internet connection and not instance type. 
    __________
    parameters 
    - instance : dict. Response dictionary from ec2 instance describe_instance method 
    - user_name : string. SSH username for accessing instance, default usernames for AWS images can be found at https://alestic.com/2014/01/ec2-ssh-username/
    - get : str or list of str. File or list of file paths to get from the instance 
    - put : str or list of str. Folder to place the files in `get` 
    '''

    if kp_dir is None:
        kp_dir = sutils.get_default_kp_dir()

    client = boto3.client('ec2', region_name='us-east-1')
    client = ec2_methods.connect_to_instance(
        instance['PublicDnsName'], kp_dir + '/' + instance['KeyName'], username=username, port=22)

    stfp = client.open_sftp()

    for idx, file in enumerate(get):
        try:
            stfp.get(file, put[idx], callback=sutils.printTotals)
        except Exception as e:
            print(file)
            raise e
    return True


def terminate_instance(instance_id):
    '''Terminate  an instance using the instance ID'''

    if type(instance_id) is str:
        instances = [instance_id]

    elif type(instance_id) is list:
        instances = instance_id

    else:
        raise Exception('instance_id arg must be str or list')

    ec2 = boto3.resource('ec2')
    ec2.instances.filter(InstanceIds=instances).terminate()
