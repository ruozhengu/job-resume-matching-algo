# import packages
import requests
import pandas as pd
import time 
from functions import *


# limit per sity
max_results_per_city = 990

# db of city 
#city_set = ['Toronto', 'Markham', 'North+York', 'Scarbough', 'Missisauga', 'Richmond+Hill']
city_set = ['Toronto', 'Vancouver', 'alberta', 'waterloo', 'london']
# job roles
job_set = ['all']


df = pd.DataFrame(columns = ['unique_id', 'job_title', 'company_name', 'location', 'full_text'])


# loop on all cities
for city in city_set:

    print("Start searching city: " + city)
    cnt = 0
                  
    # for results
    for start in range(0, max_results_per_city, 10):

        # get dom 
        page = requests.get('http://ca.indeed.com/jobs?q=&l=' + str(city) + '&sort=date&start=' + str(start))

        #ensuring at least 1 second between page grabs                    
        time.sleep(1)  

        #fetch data
        soup = get_soup(page.text)
        divs = soup.find_all(name="div", attrs={"class":"row"})
        
        # if results exist
        if(len(divs) == 0):
            break

        # for all jobs on a page
        for div in divs: 

            cnt += 1

            #specifying row num for index of job posting in dataframe
            num = (len(df) + 1) 

            #job data after parsing
            job_post = [] 

            #append unique id
            job_post.append(div['id'])

            #grabbing job title
            job_post.append(extract_job_title(div))

            #grabbing company
            job_post.append(extract_company(div))

            #grabbing location name
            job_post.append(extract_location(div))

            #grabbing link
            link = extract_link(div)

            #grabbing full_text
            job_post.append(extract_fulltext(link))

            #appending list of job post info to dataframe at index num
            df.loc[num] = job_post

    print(city + " is completed with count as " + str(cnt))
        
#saving df as a local csv file 
df.to_csv('jobs.csv', encoding='utf-8')
    
