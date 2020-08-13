# Attack Category
# ATTACK_CATEGORY = 'DOS'
# ATTACK_CATEGORY = 'U2R_AND_R2L'
# Blackbox IDS Model
# IDS_MODEL = 'SVM'

# IDS_MODELS = ['Blackbox_IDS', 'SVM', 'MLP']

# # ---- LABEL TRAFFIC OF A SPECIFIC ATTACK CATEGORY------
# # List Label of DoS Traffic
# DOS = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'mailbomb', 'processtable', 'udpstorm', 'apache2', 'worm']
# # List Label of Probe Traffic
# PROBE = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
# # List Label of U2R Traffic
# U2R = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']
# # List Label of R2L Traffic
# R2L = ['ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 'spy', 'snmpguess', 'warezclient', 'warezmaster', 'xlock', 'xsnoop']
# # List Label of U2R and R2L Traffic
# U2R_AND_R2L = U2R + R2L


# ---- FEATURE INDEX ----
INTRINSIC_INDEX = list(range(0,9))    # 0 - 8
CONTENT_INDEX = list(range(9, 23))    # 9 - 22
TIMEBASED_INDEX = list(range(23, 31)) # 23 - 30
HOSTBASED_INDEX = list(range(31,41))  # 31 - 40

DOS_FEATURES = INTRINSIC_INDEX + TIMEBASED_INDEX
U2R_AND_R2L_FEATURES = INTRINSIC_INDEX + CONTENT_INDEX