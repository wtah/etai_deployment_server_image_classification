import requests
url = 'http://127.0.0.1:8000/'
print('posting healthy image')
files = {'sample': open('sample_data/image/healthy.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())

print('\n\n')
print('posting bean image')
files = {'sample': open('sample_data/image/bean.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())

print('\n\n')
print('posting angular image')
files = {'sample': open('sample_data/image/angular.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())