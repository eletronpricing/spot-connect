
"""
Author: Carlos Valcarcel <carlos.d.valcarcel.w@gmail.com>

This file is part of spot_aws 

Launch and connect to spot instances

Examples: 
  Spot instance launch from windows command prompt: 
       $ python spot_connect -n test -p t3.micro 
    
Notes: 
  <configuration>: the aws client has already been configured using the awscli through the anaconda prompt.   
                   To do this `pip install awscli` and from the anaconda (or other python prompt) run `aws config` and follow the prompts.  
    
  <installing non-native packages on instances>: Use the scripts argument to submit bash scripts that can install non-native requirements automatically.

MIT License
"""

from path import Path
import time
import os
import sys

root = Path(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, root)

import sutils
import ec2_methods
import iam_methods
import efs_methods
import instance_methods
import bash_scripts


profiles = sutils.load_profiles()


def launch_instance(name='', instanceid='', keypair='', keypair_dir='', securitygroup='', instanceprofile='', profilespot=list(profiles.keys())[0], choices=profiles.keys(), efsmount=False, firewall='', imageid='', price='', region='us-east-1', zone='a', script='', username='ubuntu', filesystem='', newmount=False, upload='', remotepath='', activeprompt=False, terminate=False, monitoring=True):

    profile = profiles[profilespot]

    if instanceid != '':
        spot_identifier = instanceid
        using_id = True
    elif name == '':
        raise Exception("Must submit a name <-n> or instance id <-iid>.")
    else:
        spot_identifier = name
        using_id = False

    print('', flush=True)
    print('#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#', flush=True)
    print('#~#~#~#~#~#~#~# Spotting ' + spot_identifier, flush=True)
    print('#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#', flush=True)
    print('', flush=True)

    if keypair != '':
        profile['key_pair'] = (keypair, keypair + '.pem')

    if filesystem != '':
        print('Instance will be mounted on the ' +
              filesystem + ' elastic filesystem')
        profile['efs_mount'] = True
    elif filesystem == '':
        print('No EFS mount requested for this instance.')
        profile['efs_mount'] = False

    if firewall != '':
        profile['firewall_ingress'] = firewall

    if imageid != '':
        profile['image_id'] = imageid

    if price != '':
        profile['price'] = price

    if username != '':
        profile['username'] = username

    if region != '':
        profile['region'] = region

    if zone != '':
        profile['zone'] = zone

    if securitygroup != '':
        # Retrieve the security group
        sg = iam_methods.retrieve_security_group(
            securitygroup, region=profile['region'])
        # For the profile we need a tuple of the security group ID and the security group name.
        profile['security_group'] = (sg['GroupId'], securitygroup)

    if keypair_dir == '':

        try:
            kp_dir = sutils.get_package_kp_dir()
            if kp_dir == '':
                raise Exception
            print('Default key-pair directory is "%s"' % kp_dir)
        except:
            kp_dir = input(
                'Please select a default directory in which to save your key-pairs: ')
            sutils.set_default_kp_dir(kp_dir)
            print('You can change the default key-pair directory using spot_connect.sutils.set_default_kp_dir(<dir>)' % kp_dir)

    else:
        kp_dir = keypair_dir

    # Add a forward slash to the kp_dir
    if kp_dir[-1] != '/':
        kp_dir = kp_dir + '/'

    # Launch the instance using the name profile, instance profile and monitoring arguments
    try:
        # If a key pair and security group were not added provided, they wil be created using the name of the instance
        instance, profile = ec2_methods.get_spot_instance(spot_identifier, profile, instance_profile=instanceprofile, monitoring=monitoring,
                                                          kp_dir=kp_dir, using_instance_id=using_id)  # Launch or connect to the spot instance under the given name
    except Exception as e:
        raise e

    # If a filesystem was provided and we want to mount an EFS
    if profile['efs_mount']:

        print('Requesting EFS mount...')
        fs_name = filesystem
        try:                                                               # Create and/or mount an EFS to the instance
            mount_target, instance_dns, filesystem_dns = efs_methods.retrieve_efs_mount(
                fs_name, instance, new_mount=newmount, region=profile['region'])
        except Exception as e:
            raise e

        print('Connecting to instance to link EFS...')
        instance_methods.run_script(instance, profile['username'], bash_scripts.compose_mount_script(
            filesystem_dns), kp_dir=kp_dir, cmd=True)

    st = time.time()

    if upload != '':
        files_to_upload = []

        try:
            for file in upload.split(','):
                files_to_upload.append(os.path.abspath(file))
            instance_methods.upload_to_ec2(
                instance, profile['username'], files_to_upload, remote_dir=remotepath)

            print('Time to Upload: %s' % str(time.time() - st))
        except Exception as e:
            print(e)
            print("An error ocurred while trying to upload to remote path")

    st = time.time()

    scripts_to_run = []
    if script != '':
        for s in script.split(','):
            scripts_to_run.append(s)

    for script in profile['scripts'] + scripts_to_run:
        print('\nExecuting script "%s"...' % str(script))
        try:
            if not instance_methods.run_script(instance, profile['username'], script):
                break
        except Exception as e:
            print(str(e))
            print('Script %s failed with above error' % script)

        print('Time to Run Script: %s' % str(time.time() - st))

    if activeprompt:
        instance_methods.active_shell(instance, profile['username'])

    # If we want to terminate the instance
    if terminate:
        # termination overrrides everything else
        instance_methods.terminate_instance(instance['InstanceId'])
        print('Instance %s has been terminated' % str(spot_identifier))

    return instance


if __name__ == "__main__":
    launch_instance(name='teste15', profilespot='t2.micro')
