#%%
import time
import os
import re
import numpy as np
import pandas as pd
from pandas.core.indexes.base import Index

from tqdm import tqdm

#%%
# !pip install chromedriver_autoinstaller
import selenium
import chromedriver_autoinstaller
from selenium import webdriver
from selenium.webdriver import ActionChains

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, ElementNotInteractableException

#%%
url = 'https://www.csi.go.kr/acd/acdCaseList.do'

option = webdriver.ChromeOptions()
option.add_argument('window-size=1920,1080')

driver_path = chromedriver_autoinstaller.install()
driver = webdriver.Chrome(driver_path, options = option)
driver.implicitly_wait(5)
driver.get(url)

#%%
page_path = '//*[@id="main"]/div/div[2]/div/div/div'
page_length_tmp = driver.find_element_by_xpath(page_path).text
page_length = int(page_length_tmp.split(' ')[3])


accid_dict = []
start = time.time()
for j in range(1, page_length):
    accid_num_path = '//*[@id="main"]/div/div[1]/table/tbody'
    accid_num_tmp = driver.find_element_by_xpath(accid_num_path)
    accid_num = len(accid_num_tmp.find_elements_by_tag_name('tr'))

    row_length = tqdm(range(1, accid_num + 1), file = sys.stdout)

    for i in row_length:
        accid_path = '/html/body/div[3]/div[3]/div/div/div[3]/section/form/div/div[1]/table/tbody/tr[%d]/td[2]/a' % i
        accid_url_path = driver.find_element_by_xpath(accid_path)
        accid_url = accid_url_path.get_attribute('href')
        driver.execute_script(accid_url)
        time.sleep(1)
        
        # ------------------------------------------------------------------------------------------------------------- #
        tab_path = '/html/body/div[3]/div[3]/div/div/div[3]/section/form[2]/div[1]/table'
        tab_tmp = driver.find_element_by_xpath(tab_path)
        tab = tab_tmp.find_elements_by_tag_name('tr')
        
        dict_tmp = dict()
        
        # 사고명
        name = tab[0].find_element_by_class_name('td-head').text
        name_val = tab[0].find_element_by_class_name('t-left').text
        dict_tmp[name] = name_val

        # 발생일시
        date = tab[1].find_elements_by_class_name('td-head')[0].text
        date_val = tab[1].find_elements_by_class_name('t-left')[0].text
        dict_tmp[date] = date_val

        # 기상상태
        weather_tmp = tab[2].find_elements_by_class_name('t-left')[-1].text.split(' ')
        weather = weather_tmp[0]; weather_val = weather_tmp[2]
        temp = weather_tmp[3]; temp_val = weather_tmp[5]
        wet = weather_tmp[6]; wet_val = weather_tmp[-1]
        dict_tmp[weather] = weather_val
        dict_tmp[temp] = temp_val
        dict_tmp[wet] = wet_val
        
        # 사고유형
        accid_type1 = tab[4].find_elements_by_class_name('td-head')[1].text
        accid_type_val1 = re.sub('\(.*?\)', '', tab[4].find_element_by_class_name('t-left').text)
        dict_tmp[accid_type1] = accid_type_val1

        accid_type2 = tab[5].find_element_by_class_name('td-head').text
        accid_type_val2 = tab[5].find_element_by_class_name('t-left').text
        dict_tmp[accid_type2] = accid_type_val2

        try: 
            # 공종
            gz_val = tab[6].find_element_by_class_name('t-left').text.split(' > ')
            dict_tmp['공종1'] = gz_val[0]
            dict_tmp['공종2'] = gz_val[1]
        except IndexError:
            dict_tmp['공종1'] = np.nan
            dict_tmp['공종2'] = np.nan

        # 작업프로세스
        process = tab[8].find_element_by_class_name('td-head').text
        process_val = tab[8].find_element_by_class_name('t-left').text
        dict_tmp[process] = process_val

        # 사고경위
        detail = tab[11].find_element_by_class_name('td-head').text
        detail_val = tab[11].find_element_by_class_name('t-left').text
        dict_tmp[detail] = detail_val

        # 사고원인
        cause = tab[12].find_element_by_class_name('td-head').text
        cause_val = tab[12].find_element_by_class_name('t-left').text
        dict_tmp[cause] = cause_val

        # 구체적 사고원인
        specific_cause = tab[13].find_element_by_class_name('td-head').text
        specific_cause_val = tab[13].find_element_by_class_name('t-left').text
        dict_tmp[specific_cause] = specific_cause_val
        
        accid_dict.append(dict_tmp)
        
        driver.back()
        
        row_length.set_postfix({'page': j})
    
    if j % 10 == 0:
        next_page_path = '/html/body/div[3]/div[3]/div/div/div[3]/section/form/div/div[2]/div/ul/li[13]/a'
        driver.find_element_by_xpath(next_page_path).click()
        time.sleep(1)
        
    elif j % 10 != 0: 
        p = j % 10
        page_path = '/html/body/div[3]/div[3]/div/div/div[3]/section/form/div/div[2]/div/ul'
        page_list = driver.find_element_by_xpath(page_path).find_elements_by_tag_name('li')[2:-2]
        page_list[p].find_element_by_tag_name('a').click()
        time.sleep(1)
    
print('\ntime: ', time.time() - start)


#%%
accident = pd.DataFrame(accid_dict)
accident = accident.sort_values('발생일시', ascending = False).reset_index(drop = True)
accident.to_csv('accident.csv', header = True, index = False, encoding = 'utf-8-sig')