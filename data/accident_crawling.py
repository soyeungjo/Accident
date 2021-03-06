#%%
import time
import re
import openpyxl
import numpy as np
import pandas as pd
from pandas.core.indexes.base import Index

from tqdm import tqdm

#%%
# !pip install chromedriver_autoinstaller
import selenium
from bs4 import BeautifulSoup
import chromedriver_autoinstaller
from selenium import webdriver
from selenium.webdriver import ActionChains

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, ElementNotInteractableException

# from html_table_parser import parser_functions as parser


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
page_length = int(re.sub(',', '', page_length_tmp.split(' ')[3]))


result_ = []
start = time.time()
for j in range(1, page_length):
    num_path = '//*[@id="main"]/div/div[1]/table/tbody'
    num_tmp = driver.find_element_by_xpath(num_path)
    num = len(num_tmp.find_elements_by_tag_name('tr'))

    row_length = tqdm(range(1, num + 1), file = sys.stdout)

    for i in row_length:
        accid_dict = {}
        
        accid_num_path = '//*[@id="main"]/div/div[1]/table/tbody/tr[%d]/td[1]' % i
        accid_num = driver.find_element_by_xpath(accid_num_path).text
        accid_dict['accid_num'] = accid_num
        
        accid_path = '/html/body/div[3]/div[3]/div/div/div[3]/section/form/div/div[1]/table/tbody/tr[%d]/td[2]/a' % i
        accid_url = driver.find_element_by_xpath(accid_path).get_attribute('href')
        driver.execute_script(accid_url)
        time.sleep(0.5)
        
        
        # -------------------------------- ???????????? -------------------------------- #
        tab = driver.find_elements_by_class_name('table.table-bordered')
        case_tab = tab[0]
        
        not_head = ['????????????', '????????????', '????????????', '????????????']
        case_head = [x.text for x in case_tab.find_elements_by_class_name('td-head') if x.text not in not_head]
        case_val = [x.text for x in case_tab.find_elements_by_class_name('t-left')]
        
        case_tmp = {case_head[i]: case_val[i] for i in range(len(case_head))}
        accid_dict.update(case_tmp)
        
        # -------------------------------- ???????????? -------------------------------- #
        char_tab = tab[1]
        
        char_head = [x.text for x in char_tab.find_elements_by_class_name('td-head')]
        char_val = [x.text for x in char_tab.find_elements_by_class_name('t-left')]
        
        char_tmp = {char_head[i]: char_val[i] for i in range(len(char_head))}
        accid_dict.update(char_tmp)
        
        result_.append(accid_dict)
        
        driver.back()
        time.sleep(0.5)
        
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
accident = pd.DataFrame(result_)

accident.to_excel('accident.xlsx', header = True, index = False)