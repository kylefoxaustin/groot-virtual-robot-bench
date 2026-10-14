"""Recreate the deploy's reference/example motions on tmpfs so that directory iteration order == the RTX 5090 load
order (the first motion loaded sets the stand reference, and the keyboard script selects the macarena clip by index).
Verifies by listing; prints ORDER_OK or ORDER_FAIL. Run as root on the Jetsons (systemd RemoveIPC wipes a user's
/dev/shm files at logout).   usage: mkref.py <gear_sonic_deploy>/reference/example"""
import os, shutil, sys
SRC = sys.argv[1]
DST = '/dev/shm/sonic_ref/example'
WANT = ['dance_in_da_party_001__A464', 'macarena_001__A545_M', 'tired_one_leg_jumping_R_001__A359_M', 'dance_in_da_party_001__A464_M',
        'squat_001__A359', 'macarena_001__A545', 'tired_one_leg_jumping_R_001__A359', 'walking_quip_360_R_002__A428',
        'neutral_kick_R_001__A543', 'forward_lunge_R_001__A359_M', 'tired_forward_lunge_R_001__A359_M', 'neutral_kick_R_001__A543_M',
        'walking_quip_360_R_002__A428_M']
assert sorted(WANT) == sorted(os.listdir(SRC))
def build(order):
    shutil.rmtree('/dev/shm/sonic_ref', ignore_errors=True); os.makedirs(DST)
    for d in order: shutil.copytree(f'{SRC}/{d}', f'{DST}/{d}')
    return [e.name for e in os.scandir(DST)]
got = build(WANT)
if got != WANT:
    got = build(WANT[::-1])
print('ORDER_OK' if got == WANT else 'ORDER_FAIL', got)
