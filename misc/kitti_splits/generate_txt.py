with open('test_copy.txt','a') as f:
    for i in range(144):
        f.write('2011_09_26/2011_09_26_drive_0013_sync '+'{} '.format(str(i).zfill(10))+'l'+'\n')