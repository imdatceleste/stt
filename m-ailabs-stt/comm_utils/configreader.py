# -*- coding: utf-8 -*-
import sys 
reload(sys)
sys.setdefaultencoding("utf-8")
import os
import ConfigParser
    
def getKeyFromSectionInConfiguration(section, key, defaultValue, configDictionary):
    if section in configDictionary:
        val = configDictionary[section]
        if key in val:
            return val[key]

    return defaultValue

    
def getSectionFromConfiguration(section, defaultValue, configDictionary):
    if section in configDictionary:
        return configDictionary[section]
    else:
        return defaultValue


def safe_create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def getConfiguration(configFile):
    configuration = {}
    def getConfigOptionsForSection(config, section):
        retDict = {}
        for item in config.items(section):
            retDict[item[0]] = config.get(section, item[0])
        return retDict

    def extractConfigOptionsForSection(config, section):
        retDict = getConfigOptionsForSection(config, section)
        if retDict:
            configuration[section] = retDict

    if configFile:
        config = ConfigParser.ConfigParser()
        try:
            config.readfp(open(configFile))
        except:
            print("Confing File <", configFile, "> could not be read")
            sys.exit(1)
    else:
        print("Please provide a config file with the appropriate option")
        sys.exit(1)

    for section in config.sections():
        extractConfigOptionsForSection(config, section)

    return configuration


