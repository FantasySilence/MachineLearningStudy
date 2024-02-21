import os


class FilesIO:

    """
    这是一个文件IO流类
    用于数据的读取
    """

    @staticmethod
    def getDataRoot():

        """
        获取数据集根路径
        """

        rootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataPath = os.path.join(rootPath, "resources")
        return dataPath
    

    @staticmethod
    def getLectureData(file_name: str):

        """
        获取课程数据
        """

        dataRootPath = FilesIO.getDataRoot()
        dataPath = os.path.join(dataRootPath, "lecturedata", file_name)
        return dataPath
    

    @staticmethod
    def getHomeworkData(file_name: str):

        """
        获取作业数据
        """

        dataRootPath = FilesIO.getDataRoot()
        dataPath = os.path.join(dataRootPath, "homeworkdata", file_name)
        return dataPath

if __name__ == "__main__":
    
    print(FilesIO.getLectureData("IRIS.csv"))
    print(FilesIO.getHomeworkData("IRIS.csv"))