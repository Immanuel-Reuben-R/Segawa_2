import getpass
import os

env_path = os.path.expanduser('~/.env')
print("--- Segawa Secure Credential Setup ---")
email = input('Enter your Gmail Address: ')
password = getpass.getpass('Enter your App Password (typing will be hidden): ')

with open(env_path, 'a') as f:
    f.write(f'\nSENDER_EMAIL={email}\n')
    f.write(f'SENDER_PASSWORD={password}\n')

print("Credentials saved successfully to ~/.env!")
