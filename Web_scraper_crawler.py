import requests
import pprint
import json
from bs4 import BeautifulSoup
import urllib
import pickle
import regex as re

def cleans_dictionary(raw_dictionary):
 
    post_dist = {}
    global new_submissions
    for post in raw_dictionary['topic_list']['topics'] :
      
      post_dist['id'] = str(post['id'])
      post_dist['slug'] = str(post['slug'])
      post_dist['tags'] = post['tags']
      new_submissions.append(post_dist)
      post_dist={}
    #print(raw_dictionary)


def get_forum_message(page):    
    
    soup = BeautifulSoup(page.content, 'html.parser')

    if not re.findall(r"\- (.*?)\- BNZ Community",str(soup.find('title'))) : # marks category to " Other " incase of posts missing category (changes with forum)
      category = 'Other'
    else:
      title = str(soup.find('title')).split('-')
      category = title[-2]#changes with forum
    

    for anchors in soup.find_all('a'): #gets rid of hyperlinks embedded in text
      anchors.extract()

    postData = soup.find_all("p")
    
    posts = []
    for post in postData:
        posts.append(BeautifulSoup(str(post)).get_text().strip('').replace("\r",""))

    posts_stripped = [x.replace("\n","") for x in posts]  
    
    return category,''.join(posts_stripped)

    

#page_amount = 1
new_submissions = []
clean_posts=[]
#for page in range(1,page_amount+1) :
page=0
while True:
    page=page+1
    url = "https://community.bnz.co.nz/latest.json?no_definitions=true&page="+str(page)
    r = (requests.get(url)).text
    raw_dictionary = json.loads(r)
    if len(raw_dictionary['topic_list']['topics']) <= 0:
      print('Done')
      break
    if (page%5==0):
      print("Page",page)
    cleans_dictionary(raw_dictionary)
    
with open("bnz_slug.pickle", "wb") as output_file:
  pickle.dump(new_submissions, output_file)

topic_url = "https://community.bnz.co.nz/t/"
print("Total Topics:",len(new_submissions))
for submission in new_submissions:

    topic_url += (submission['slug'] + '/' + submission['id'])
    page = requests.get(topic_url)
    category,text = get_forum_message(page)

    # url = topic_url+'.json'
    # r1= (requests.get(url)).text  
    
    # raw = (json.loads(r1))
    # tags = raw['tags']

    if submission['tags'] ==[]:
      submission['tags'] = ['NA']

    clean_posts.append([topic_url,category,submission['tags'],text])
    topic_url = "https://community.bnz.co.nz/t/"

with open("Data.pickle", "wb") as output_file:
    pickle.dump(clean_posts, output_file)


# returns data in [url,category,[tags],text] format
