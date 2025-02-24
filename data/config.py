import os
import yaml

from box import Box
from typing import List, Dict
from collections import OrderedDict


class Config(object):
    """Config class for reading configs/config.yaml
    """

    def __init__(self,
                 yaml_path: str = '') -> None:
        """
        """
        self._repo_yaml = "configs/config.yaml"
        if yaml_path:
            self.yaml_path = yaml_path
        else: 
            self.yaml_path = self._get_default_yaml()
        self._init()
    
    def _init(self) -> None:
        """
        """
        _config = Box(self._load_config())
        for key in _config.keys():
            setattr(self, key, _config[key])

    def _load_config(self) -> Dict:
        """
        """
        with open(self.yaml_path) as f:
            _config = yaml.safe_load(f)
        return _config
    
    def _get_default_yaml(self):
        """
        """
        script_dir = os.path.dirname(__file__)
        repo_dir = script_dir[:script_dir.rfind('/')]
        return f'{repo_dir}/{self._repo_yaml}'

    def get_datapath(self,
                     data: str) -> str:
        """
        """
        _data = data.lower()
        assert _data in self.datasets

        if _data in self.data.repo:
            dpath = os.path.join(
                        self.root.repo, 
                        self.data.repo[_data])
        else:
            dpath = os.path.join(
                        self.root.vcluster,
                        self.data.vcluster[_data])
        return dpath

    def get_datafiles(self,
                      data: str,
                      exclude_subdirs: bool = True) -> OrderedDict:
        """
        """
        datapath = self.get_datapath(data)
        datafiles = OrderedDict()
        for root, dirs, files in os.walk(datapath):
            datafiles.update(
                    [
                        (fname, os.path.join(root, fname))
                        for fname in files
                    ]
                )
            datafiles.update(
                    [
                        (dirname, os.path.join(root, dirname))
                        for dirname in dirs
                    ]
                )
            if exclude_subdirs:
                break
        return datafiles
    
    @property
    def datasets(self) -> List:
        """
        """
        #assert 'data' in self.__dict__.keys()
        return [*self.data.repo] + [*self.data.vcluster]
    
    def __repr__(self):
        repr = ""
        for key in self.__dict__.keys():
            repr += f"{key}: {self.__dict__[key]}\n"
        return repr