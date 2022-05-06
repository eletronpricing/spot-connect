"""
Author: Carlos Valcarcel <carlos.d.valcarcel.w@gmail.com>

This file is part of spot-connect

Toolbox for working with AWS ec2-instances:

the sutils submodule contains general-use python functions that are needed to
build out the other sub-modules in spot-connect.

MIT License 2020
"""

import os
import ast
import boto3
import random
import string
import pprint
import glob
import re
import psutil
import _pickle as pickle
import pandas as pd
import numpy as np
from path import Path
from IPython.display import clear_output
from datetime import datetime, timedelta
from tabulate import tabulate

root = Path(os.path.dirname(os.path.abspath(__file__)))


def full_pickle(title, data):
    '''pickles the submited data and titles it'''
    pikd = open(title + '.pickle', 'wb')
    pickle.dump(data, pikd)
    pikd.close()


def loosen(file):
    '''loads and returns a pickled objects'''
    pikd = open(file, 'rb')
    data = pickle.load(pikd)
    pikd.close()
    return data


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def genrs(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def absoluteFilePaths(directory):
    '''Get the absolute file path for every file in the given directory'''

    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def pull_root():
    '''Retrieve the directory for this instance'''
    return Path(os.path.dirname(os.path.abspath(__file__)))


def load_profiles():
    '''Load the profiles from the package profile.txt file'''

    profile = [f for f in list(absoluteFilePaths(
        pull_root() + '/data/')) if f.split('\\')[-1] == 'profiles.txt'][0]

    with open(profile, 'r') as f:
        profiles = ast.literal_eval(f.read())

    # print('Profiles loaded, you can edit profiles in '+str(profile))

    return profiles


def save_profiles(profiles):
    '''Save the profile dict str in a .txt file'''
    profile_file = [f for f in list(absoluteFilePaths(
        pull_root() + '/data/')) if f.split('\\')[-1] == 'profiles.txt'][0]

    # ptosave = ast.literal_eval(profile_str)
    print(profile_file)

    with open(profile_file, 'w') as f:
        f.write(pprint.pformat(profiles))
        f.close()


def default_region():
    profiles = load_profiles()
    print(profiles['default']['region'])


def change_default_region(region, deactive_warning=True):
    if not deactive_warning:
        ans = input(
            'Warning: doing this will change the "region" for all profiles. Continue?(y): ')
        if ans != 'y':
            raise Exception('User exit')

    profiles = load_profiles()
    for k in profiles:
        profiles[k]['region'] = region

    save_profiles(profiles)


def change_default_image(image, deactive_warning=True):
    if not deactive_warning:
        ans = input(
            'Warning: doing this will change the "image_id" for all profiles. Continue?(y): ')
        if ans != 'y':
            raise Exception('User exit')

    profiles = load_profiles()
    for k in profiles:
        profiles[k]['image_id'] = image

    save_profiles(profiles)


def show_instances():
    client = boto3.client('ec2', region_name='us-east-1')
    print('Instances (by Key names):')
    for i in [res['Instances'][0] for res in client.describe_instances()['Reservations']]:
        print('     - "' + i['KeyName'].split('-')[1] + '" Type: ' +
              i['InstanceType'] + ', ID: ' + i['InstanceId'], flush=True)


def list_instance_profiles():
    '''List all instance profile roles avaialable. Instance profiles assign roles to instances so they can access other AWS services like S3.'''
    iam_client = boto3.client('iam')
    return iam_client.list_instance_profiles()


def printTotals(transferred, toBeTransferred):
    '''Print paramiko upload transfer'''
    print("Transferred: %.3f" % float(float(transferred) /
          float(toBeTransferred)), end="\r", flush=True)


def get_package_kp_dir():
    '''Get the key-pair directory'''
    kpfile = [f for f in list(absoluteFilePaths(
        pull_root() + '/data/')) if f.split('\\')[-1] == 'key_pair_default_dir.txt'][0]
    with open(kpfile, 'r') as f:
        default_path = f.read()
        f.close()
    return default_path


def get_default_kp_dir():
    '''Get the default key pair directory'''
    kp_dir = get_package_kp_dir()
    return kp_dir


def set_default_kp_dir(directory: str):
    '''Set the default key pair directory'''
    kpfile = [f for f in list(absoluteFilePaths(
        pull_root() + '/data/')) if f.split('\\')[-1] == 'key_pair_default_dir.txt'][0]
    with open(kpfile, 'w') as f:
        f.write(directory)
        f.close()
    print('Default path has been set to ' + kpfile)


def clear_key_pairs():
    '''Erase all the key pairs in the kp_directory'''
    answer = input('You are about to erase all the locally stored key pairs.\nYou will have to erase the matching key board through the AWS dashboard. Conitnue? (Y)')

    if answer == 'Y':
        for f in glob.glob(get_default_kp_dir() + '/*'):
            os.remove(f)
    else:
        raise Exception('User exit')


def find_username(s):
    for k in username_dictionary:
        ios = re.findall('(' + k + ')', s)
        if len(ios) > 0:
            ios = username_dictionary[ios[0]]
            break
    return ios


def select_region():

    results = []

    for i, r in enumerate(ami_data['region'].unique()):
        # print(i, r)
        results.append({
            'indice': i,
            'regiao': r
        }
        )

    df = pd.DataFrame(results)

    check_indice = True

    while check_indice:
        try:

            print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))
            region_idx = int(
                input('Insira o indice da regiao desejada: '))
            region = list(ami_data['region'].unique())[region_idx]
            clear_output()
            check_indice = False
        except Exception:
            print('\nindice nao localizado. Tente novamente.\n')

    return region


def select_instance_type(region='us-east-1'):

    results = []

    instance_list = spot_instance_pricing.loc[spot_instance_pricing['region']
                                              == region, 'instance_type']
    for i, r in enumerate(instance_list):
        # print(i, r, '(' + str(list(
        #     spot_instance_pricing.loc[spot_instance_pricing['region'] == region, 'vcpus'])[i]) + ')')

        results.append({
            'indice': i,
            'instancia': r,
            'cpu': list(spot_instance_pricing.loc[spot_instance_pricing['region'] == region, 'vcpus'])[i]
        }
        )

    df = pd.DataFrame(results)

    check_indice = True

    while check_indice:

        try:
            print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))
            instance_idx = int(
                input('Insira o indice da instancia desejada: '))
            instance_cpu = list(spot_instance_pricing.loc[spot_instance_pricing['region'] == region, 'vcpus'])[
                instance_idx]
            instance_name = list(spot_instance_pricing.loc[spot_instance_pricing['region'] == region, 'instance_type'])[
                instance_idx]
            clear_output()
            check_indice = False
        except Exception:
            print('\nindice nao localizado. Tente novamente.\n')

    return instance_name, instance_cpu


def select_image(region='us-east-1'):

    results = []

    image_list = ami_data.loc[ami_data['region'] == region, 'image_name']

    for i, r in enumerate(image_list):

        results.append({
            'indice': i,
            'imagem': r
        }
        )

    df = pd.DataFrame(results)

    check_indice = True

    while check_indice:
        try:

            print(tabulate(df, headers='keys',
                           tablefmt='pretty', showindex=False))
            image_idx = int(
                input('Insira o indice da imagem desejada: '))
            image_id = list(ami_data.loc[ami_data['region'] == region, 'image_id'])[
                image_idx]
            image_name = list(ami_data.loc[ami_data['region'] == region, 'image_name'])[
                image_idx]
            username = list(ami_data.loc[ami_data['region'] == region, 'username'])[
                image_idx]
            clear_output()
            check_indice = False
        except Exception:
            print('\nindice nao localizado. Tente novamente.\n')

    return image_id, image_name, username


def add_profile(profile_dict, instance_type, image_id, image_name, bid_price, min_price, no_cpus, region='us-east-1', zone='a', username='ubuntu'):
    profile_dict[instance_type] = {
        'efs_mount': str(False),
        'firewall_ingress': ('tcp', 22, 22, '0.0.0.0/0'),
        'image_id': image_id,
        'image_name': image_name,
        'instance_type': str(instance_type),
        'price': str(bid_price),
        'min_price': str(min_price),
        'vcpus': str(no_cpus),
        'region': str(region),
        'zone': str(zone),
        'scripts': [],
        'username': str(username)
    }
    return profile_dict


def reset_profiles(price_increase=1.15):
    '''Reset the profile image, region, and set what % of the price you want to set as maximum bid for all instance types (remember you can always submit a custom price when making spot-requests).'''
    assert price_increase >= 1

    region = select_region()
    image_id, image_name, username = select_image(region)

    # region_name = region.split(')')[0] + ')'
    region_code = region  # .split(')')
    zone_code = 'a'
    spot_instance_pricing.loc[spot_instance_pricing['region'] == region_code]

    profile_dict = {}
    for tup in spot_instance_pricing.itertuples():

        if 'N/A' in tup.linux_price:
            continue
        try:
            instance_price = float(re.findall(
                '([0-9]*\.[0-9]*)', tup.linux_price)[0])
        except:
            instance_price = 0
        no_cpus = tup.vcpus
        bid_price = round(instance_price * price_increase, 4)

        profile_dict = add_profile(profile_dict,
                                   tup.instance_type,
                                   image_id,
                                   image_name,
                                   bid_price,
                                   instance_price,
                                   no_cpus,
                                   region_code,
                                   zone_code,
                                   username)
    save_profiles(profile_dict)


def count_cpus_by_type():
    print('Logical CPUs: %s' % str(psutil.cpu_count(logical=True)))
    print('Physical CPUs: %s' % str(psutil.cpu_count(logical=False)))


def split_workloads(n_jobs, workload, wrkdir=None, filename=None):
    '''Split the workload into n_jobs which are saved as pickle files in the wrkdir under the filename<i> for each job i
    This is meant to be used to create the upload material for distributed jobs.
    __________
    parameters
    - n_jobs : int. number of files to split the workload into
    - workload : list. The items that will be split and pickled
    - wrkdir : str. The path to store the files for each job. If no directory is submitted the working directory will be printed.
    - filename : str. The prefix of each job file, the default title is "current_workload"
    '''

    if wrkdir is None:
        wrkdir = pull_root() + '/data/'
        if not os.path.exists(wrkdir):
            try:
                os.mkdir(wrkdir)
            except Exception as e:
                print(
                    'Failed to make directory, submiting an existing directory will fix this. Otherwise you can change permissions.')
                raise e

    assert os.path.exists(wrkdir)

    if filename is None:
        filename = wrkdir + '/current_workload'
    else:
        filename = wrkdir + '/' + filename

    workload_size = int(np.ceil(len(workload)) / n_jobs)

    workload_list = [c for c in chunks(workload, workload_size)]

    wnum = 0
    filenames = []
    for work in workload_list:
        full_pickle(filename + '_' + str(wnum), work)
        filenames.append(filename + '_' + str(wnum) + '.pickle')
        wnum += 1

    return filenames


class CurrentIdLog:

    def __init__(self, logdir=None, lower_limit=0, upper_limit=9999999):
        '''
        A log class to keep track of the user and call Ids that have been used today.
        This class was designed to avoid using repeat user id numbers and call numbers which will return an error in the API.
        We reset by day because call numbers are reset in the API system after a short ammount of time.
        __________
        parameters
        - logdir : str. the directory to load or store the "current_session_ids.pickle" file which contains the IDs that have been used today so far.
        '''
        self.curid = None
        self.ids = None
        self.hd = None

        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

        date = datetime.today().date()

        if logdir is None:
            self.hd = pull_root() + '/data/'
        else:
            self.hd = logdir

        if os.path.exists(self.hd + '/current_session_ids.pickle'):
            # load any existing log
            self.ids = loosen(self.hd + '/current_session_ids.pickle')
            # if it is from a different date we reset it
            if self.ids['date'] != date:
                self.ids = {}
                # and set its date to today
                self.ids['date'] = date
        else:
            self.ids = {}
            self.ids['date'] = date

    def add_user_id(self, uid):
        '''Add a user ID to the list of unavailable IDs'''
        self.ids[uid] = []

    def set_user_id(self, uid):
        '''Set the current session ID'''
        self.curid = uid
        if uid not in self.ids:
            self.add_user_id(uid)

    def add_call_id(self, fid):
        '''Log that the current fid has been used with the current session id'''
        self.ids[self.curid].append(fid)

    def get_valid_user_id(self, verbose=False):
        '''Get a user ID that has not been used yet'''
        uid = random.randint(
            self.lower_limit, self.upper_limit)                  # propose a random integer as the new user ID
        valid = False
        while not valid:
            if uid in self.ids:                                                # if it is in the current list of used IDs
                # propose another
                uid = random.randint(self.lower_limit, self.upper_limit)
            else:
                # otherwise set it as the current ID and move on
                self.set_user_id(uid)
                # save the id dictionary for todays session
                full_pickle(self.hd + '/current_session_ids', self.ids)
                valid = True
        if verbose:
            print('Current ID is %i' % uid)

        # return the user ID for use
        return uid

    def get_valid_call_id(self, verbose=False):
        '''Get a function ID that has not been used for the current user yet'''
        fid = random.randint(
            self.lower_limit, self.upper_limit)                  # propose a random integer as the new function ID
        valid = False
        while not valid:
            # if the current user has used it already
            if fid in self.ids[self.curid]:
                # propose a new one
                fid = random.randint(self.lower_limit, self.upper_limit)
            else:
                # otherwise add it to the list and use it and move on
                self.ids[self.curid].append(fid)
                # save the id dictionary for todays session
                full_pickle(self.hd + '/current_session_ids', self.ids)
                valid = True
        if verbose:
            print('Call ID is %i' % fid)

        return fid


def get_region_prices_online(instance_type, regiao='us-east-1'):
    """printa na tela os atuais precos spot da regiao

    Args:
        instance_type ([string]): tipo da instancia
        regiao ([string]): regiao
    """

    client = boto3.client('ec2', region_name=regiao)

    print("\nInstancia: %s" % instance_type)

    results = []

    client = boto3.client('ec2', region_name=regiao)
    prices = client.describe_spot_price_history(
        InstanceTypes=[instance_type],
        ProductDescriptions=['Linux/UNIX', 'Linux/UNIX (Amazon VPC)'],
        StartTime=(datetime.now() -
                   timedelta(hours=5)).isoformat(),
        MaxResults=100
    )

    for price in prices["SpotPriceHistory"]:
        results.append((price["AvailabilityZone"],
                        price["SpotPrice"], price["Timestamp"]))

    df = pd.DataFrame(results, columns=['Regiao', 'Preco', "Timestamp"])
    df['Preco'] = df['Preco'].str[:6]
    df = df.sort_values('Regiao')

    # idx = df.groupby("Regiao")['Timestamp'].transform(
    #     max) == df['Timestamp']

    # print(tabulate(df[idx][['Regiao', 'Preco']],
    #                headers='keys', tablefmt='pretty', showindex=False))

    return df


def select_availability_zone_by_price(instance_type, regiao='us-east-1'):

    df = get_region_prices_online(instance_type, regiao)

    idx = df.groupby("Regiao")['Timestamp'].transform(
        max) == df['Timestamp']

    df_idx = df[idx][['Regiao', 'Preco']]
    lista_zonas = ['a', 'b', 'c', 'd', 'f']
    check_zona = True

    while check_zona:

        print(tabulate(df[idx][['Regiao', 'Preco']],
                       headers='keys', tablefmt='pretty', showindex=False))

        azone_code = input('Insira a letra da zona desejada: ')

        if azone_code in lista_zonas:
            regiao_az = regiao + azone_code
            preco = df_idx.loc[df_idx['Regiao']
                               == regiao_az, 'Preco'].values[0]
            return azone_code, preco
        else:
            print(f'\nA zona deve ser uma letra no intervalo {lista_zonas}\n')


def print_ami_images_online(regiao='us-east-1'):
    """printa na tela uma lista das imagens disponiveis na regiao informada

    Args:
        regiao (str, optional): string com nome da regiao que contem as imagens. Defaults to 'us-east-1'.
    """

    ec2_client = boto3.client('ec2', region_name=regiao)
    images = ec2_client.describe_images(Owners=['self'])

    lista_filtro = ['DECOMP', 'NEWAVE']

    results = []

    for image in images["Images"]:
        for item in lista_filtro:
            if item in image["Name"]:
                results.append((image["Name"],
                                image["ImageId"]))

    df = pd.DataFrame(results, columns=['Name', 'ImageId'])
    df = df.sort_values('Name')

    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))


def update_ami_images(regiao='us-east-1'):
    """reseta arquivo com lista de imagens disponiveis na regiao informada

    Args:
        regiao (str, optional): string com nome da regiao que contem as imagens. Defaults to 'us-east-1'.
    """

    ami_file_path = 'spot_connect/data/ami_data.csv'

    lista_filtro = ['DECOMP', 'NEWAVE']

    ec2_client = boto3.client('ec2', region_name=regiao)
    images = ec2_client.describe_images(Owners=['self'])

    i = 0

    with open(ami_file_path, 'w') as f:
        f.write(',image_name,image_id,region\n')
        for image in images["Images"]:
            for item in lista_filtro:
                if item in image["Name"]:
                    f.write(str(i) + ',' + image["Name"] + ',' +
                            image["ImageId"] + ',' + regiao + '\n')
                    i = i + 1


def get_ec2_instance_types(region_name='us-east-1'):
    '''Yield all available EC2 instance types in region <region_name>'''
    ec2 = boto3.client('ec2', region_name=region_name)

    describe_args = {}

    while True:
        describe_result = ec2.describe_instance_types(**describe_args)

        for i in describe_result['InstanceTypes']:
            yield (i['InstanceType'], i['VCpuInfo']['DefaultVCpus'])

        if 'NextToken' not in describe_result:
            break
        describe_args['NextToken'] = describe_result['NextToken']


def get_ec2_instance_region_avgprice(instance_type, regiao='us-east-1'):
    """retorna lista com valor medio dos precos spot da regiao nos ultimos 7 dias

    Args:
        instance_type ([string]): tipo da instancia
        regiao ([string]): regiao
    """

    client = boto3.client('ec2', region_name=regiao)

    client = boto3.client('ec2', region_name=regiao)
    prices = client.describe_spot_price_history(
        InstanceTypes=[instance_type],
        ProductDescriptions=['Linux/UNIX', 'Linux/UNIX (Amazon VPC)'],
        StartTime=(datetime.now() -
                   timedelta(days=7)).isoformat(),
        MaxResults=100
    )

    i = 0
    somaprecos = 0

    for price in prices["SpotPriceHistory"]:
        somaprecos = float(price["SpotPrice"]) + somaprecos
        i = i + 1

    try:
        precomedio = round(somaprecos / i, 4)
    except:
        precomedio = 0

    return precomedio


def update_instance_list(region_name='us-east-1'):

    instance_file_path = 'spot_connect/data/spot_instance_pricing.csv'

    lista_tipos = ['c5', 'c6i.', 'm5.']

    lista_instancia = sorted(get_ec2_instance_types(region_name))

    i = 0

    with open(instance_file_path, 'w') as f:
        f.write(',instance_type,vcpus,linux_price,windows_price,region\n')
        for ec2_type in lista_instancia:
            for item in lista_tipos:
                if item in ec2_type[0] and ec2_type[1] > 40 and 'metal' not in ec2_type[0]:
                    f.write(str(i) + ',' + ec2_type[0] + ',' + str(ec2_type[1]) + ',' + '$' +
                            str(get_ec2_instance_region_avgprice(ec2_type[0], region_name)) + ',' + 'N/A*' + ',' + region_name + '\n')
                    i = i + 1


def update_instance_list_full(region_name='us-east-1'):

    instance_file_path = 'spot_connect/data/spot_instance_pricing.csv'

    lista_instancia = sorted(get_ec2_instance_types(region_name))

    i = 0

    with open(instance_file_path, 'w') as f:
        f.write(',instance_type,vcpus,linux_price,windows_price,region\n')
        for ec2_type in lista_instancia:
            f.write(str(i) + ',' + ec2_type[0] + ',' + str(ec2_type[1]) + ',' + '$' +
                    str(get_ec2_instance_region_avgprice(ec2_type[0], region_name)) + ',' + 'N/A*' + ',' + region_name + '\n')
            i = i + 1


def update_listas(region_name='us-east-1'):

    print('atualizando lista de imagens...')
    update_ami_images(region_name)

    print('atualizando lista de instancias e precos medios...')
    update_instance_list(region_name)

    print('finalizado!')


# Load the data needed for the module
username_dictionary = {'Linux': 'ec2-user',
                       'Ubuntu': 'ubuntu',
                       'Windows': 'ec2-user'}

spot_instance_pricing = pd.read_csv(
    pull_root() + '/data/spot_instance_pricing.csv')
ami_data = pd.read_csv(pull_root() + '/data/ami_data.csv')
ami_data['username'] = ami_data['image_name'].apply(lambda s: find_username(s))


if __name__ == '__main__':
    reset_profiles()
    # update_ami_images()
    # update_instance_list_full()
    # print(select_availability_zone_by_price('c6i.32xlarge'))
    # update_instance_list_full()
