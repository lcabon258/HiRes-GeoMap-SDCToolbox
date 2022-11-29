# -*- coding: utf-8 -*-
""" SDC Toolbox by Cheng-Wei Sun
Create time: 2019/04/22
Last modified: 2022/11/29
"""
HasArcpy = True
try:
    import arcpy
except ImportError:
    HasArcpy = False

import os.path as oph

class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the .pyt file)."""
        self.label = "SDC Toolbox"
        self.alias = "SDC"

        # List of tool classes associated with this toolbox
        self.tools = [SDC2shp]

from osgeo import ogr # 向量處理引擎
import osgeo.osr as osr # 作為輸入座標系統使用
import re # 檔案輸出用
import numpy as np # 回歸計算
import numpy.linalg as npli # 回歸計算
import sys
#import shapefile as shp
import os
import time
import io
import csv

NoneType = type(None)

class bedding(object):
    ''' Bedding object for 3D geometric analysis.
    Input : 
        (*) list_of_points : input a list of points in format [ [x1,y1,z1] , [x2,y2,z2] , ... ] 
    '''
    
    def __init__(self,list_of_points_of_shape) :
        self.shape_pts=list_of_points_of_shape # 輸入的點資料暫存容器，格式：[[x1,y1,z1],...]
        self.shape_np_pts=np.array(self.shape_pts) # 轉換為 array格式方便陣列計算
        self.number_of_points = len(self.shape_pts) # 點的數量
        self.q,self.r,self.rank,self.s,self.r2=self.Regression_1D() # p,residuals,rank,s,r2 ，代表計算出來的結果
        self._a,self._b,self._c=self.q[0],self.q[1],self.q[2] # z=ax+by+c 的係數
        self.strike= self.Get_Strike() # 計算走向
        self.dip,self.dd,self.dd_s= self.Get_Dip_Degree(),self.Get_Dip_Direction(),self.Get_Dip_Direction(True) # 計算傾向與傾角
        #把小數捨去
        self.strike=round(self.strike)
        self.dip=round(self.dip)
        self.r2=np.around(self.r2,decimals=2)
        # 為了釋放空間，或許未來可以把  self.shape_pts 點刪除
        #del self.shape_pts
        # 幾何中心（centroid），這邊只提供容器裝，並不在這個物件內計算。
        self.cenX = 0.
        self.cenY = 0.
        self.cenZ = 0.
        # 加入版本號
        self._libver = "20190611"
        
    def _get_x_arr(self):
        '''Return an array contain all x cordinates'''
        return np.array([pt[0] for pt in self.shape_pts ])
    def _get_y_arr(self):
        '''Return an array contain all y cordinates'''
        return np.array([pt[1] for pt in self.shape_pts ])
    def _get_z_arr(self):
        '''Return an array contain all z cordinates'''
        return np.array([pt[2] for pt in self.shape_pts ]) 
    def _vector_angle(self,v1,v2):
        '''Angle in rad. Use np.degrees() to convert to degrees.
        '''
        return np.arccos((np.dot(v1,v2)/(npli.norm(v1)*npli.norm(v2))))
    def _vector_angle2(self,v1,v2):
        co=(np.dot(v1,v2)/(npli.norm(v1)*npli.norm(v2)))
        si=(npli.norm(np.cross(v1,v2))/(npli.norm(v1)*npli.norm(v2)))
        angle=np.arctan2(si,co)
        return angle
    def _Tight_bbox_Finder(self,ShowPlot=False):
        '''
        Read 3d points and find tight bbox of the shape
        XYZarr =  a (N x 3) array contain all the points
        thida = angle between vector and (0,-1) if b < 0.
                angle between vector and (0, 1) if b > 0.
        '''
        # fig = plt.figure(dpi=100)
        # fig.canvas.set_window_title('_Tight_bbox_Finder in bedding module. Sun@2016.08')
        # fig.suptitle("Data Points", fontsize=16)
        ## Original
        # ax0 = fig.add_subplot(1,2,1) #, projection='3d'
        # ax0.set_title("Original Points")        
        # ax0.plot(self.shape_np_pts[:,0],self.shape_np_pts[:,1],'r-',label="Original Shapefile")
        ## use "scatter to plot 3D points"        
        ## Rotated        
        ##fig1 = plt.figure()
        ##fig1.suptitle("Rotated Points", fontsize=16)
        # ax1 = fig.add_subplot(1,2,2)# projection='3d'
        # ax1.set_title("Rotated Points")        
        ##        
        Quadrant = 0
        v1 = np.array(([self._a,self._b]))
        #print("v1")
        #print(v1)
        pts = self.shape_np_pts[:,0:2].T
        #print("pts")
        #print(pts)
        if self._b > 0: #Quadrant  = I or II            
            v2 = np.array(([0.,1.]))
            if self._a>0: # 1st Quadrant 
                Quadrant=1
                rota_angle = self._vector_angle(v1,v2)
            elif self._a<0: # 2nd Quadrant
                Quadrant=2
                rota_angle = -1.*self._vector_angle(v1,v2)
            rota_Matrix = np.array(([np.cos(rota_angle),-1.*np.sin(rota_angle)],[np.sin(rota_angle),np.cos(rota_angle)]))
            #print("Rota Angle : ")
            #print(rota_angle)
            #print("Rota_matrix : ")            
            #print(rota_Matrix)
            '''
            [[X1', X2' ...]   =  [[cos(a), -sin(a)]     (dot)    [[X1, X2 ...]
              Y1', Y2' ...]]      [sin(a),  cos(b)]                Y1', Y2' ...]]
            '''            
            rotated_points = np.dot(rota_Matrix,pts)
            ##
            # ax1.plot(rotated_points[0,:],rotated_points[1,:],"c-",label="Rotated Shapefile")
            ##            
            #print("rotated_points")
            #print(rotated_points)
            
            ''' Usage of "axis" arguement :
            array([[0, 1],
                   [2, 3]])
            >>> np.amax(a)           # Maximum of the flattened array
            3
            >>> np.amax(a, axis=0)   # Maxima along the first axis
            array([2, 3])
            >>> np.amax(a, axis=1)   # Maxima along the second axis
            array([1, 3])
            '''
            xymax = np.amax(rotated_points,axis=1) # index : x = 0 ; y = 1 [[Xmax,Ymax]
            xymin = np.amin(rotated_points,axis=1) # index : x = 0 ; y = 1  [Xmin,Ymin]]
            #print("xymax")
            #print(xymax)
            #print("xymin")
            #print(xymin)
            xr = xymax[0]-xymin[0]
            yr = xymax[1]-xymin[1]
            #print("xr : {}\nyr : {}".format(xr,yr))
            xratio = 0.2
            yratio = 0.4            
            '''
            tbbox : LB,LT,RB,RT 
            [[Xmin , Xmin , Xmax , Xmax ]
             [Ymin , Ymax , Ymin , Ymax ]]
            '''
            tbbox = np.array(([[xymin[0] - (xr*xratio) , xymin[0] - (xr*xratio) , xymax[0] + (xr*xratio), xymax[0] + (xr*xratio)], \
                               [xymin[1] - (yr*yratio) , xymax[1] + (yr*yratio) , xymax[1] + (yr*yratio), xymin[1] - (yr*yratio)]]))
            ## 3D : scatter 2D line : plot
            # ax1.scatter(tbbox[0,:],tbbox[1,:],c="b",marker="o",label="tbbox")
            ##
            
            #tbbox = np.vstack ((xymax,xymin)).T
            '''
            tbbox after vstack:     Transposed :
            [[xmax,ymax]           [[xmax,xmin]
             [xmin,ymin]]           [ymax,ymin]] 
            '''
            #print("After vstack")
            #print(tbbox)
            rota_angle = -1. * rota_angle
            #print("Rota angle 2")
            #print(rota_angle)
            rota_Matrix = np.array(([np.cos(rota_angle),-1.*np.sin(rota_angle)],[np.sin(rota_angle),np.cos(rota_angle)]))
            #print("rota_Matrix 2")
            #print(rota_Matrix)            
            tbbox = np.dot(rota_Matrix,tbbox)
            #print("tbbox")
            #print(tbbox)
            ##
            # ax0.scatter(tbbox[0,:],tbbox[1,:],c="g",marker="^",label="Processed bbox")
            # ##            
            # ax0.legend(loc=9)
            # ax1.legend(loc=9)
            # if ShowPlot:
                # plt.show()
            ##
            return tbbox
            
        elif self._b < 0 : #b < 0
            v2 = np.array(([0.,-1.]))
            if self._a>0: # 4st Quadrant 
                Quadrant=4
                rota_angle = -1.*self._vector_angle(v1,v2)
            elif self._a<0: # 3rd Quadrant
                Quadrant=3
                rota_angle = self._vector_angle(v1,v2)
            rota_Matrix = np.array(([np.cos(rota_angle),-1.*np.sin(rota_angle)],[np.sin(rota_angle),np.cos(rota_angle)]))
            #print("Rota Angle : ")
            #print(rota_angle)
            #print("Rota_matrix : ")            
            #print(rota_Matrix)
            rotated_points = np.dot(rota_Matrix,pts)
            ##
            # ax1.plot(rotated_points[0,:],rotated_points[1,:],"c-",label="Rotated Shapefile")
            ##   
            #print("rotated_points")
            #print(rotated_points)
            xymax = np.amax(rotated_points,axis=1) # index : x = 0 ; y = 1 [[Xmax,Ymax]
            xymin = np.amin(rotated_points,axis=1) # index : x = 0 ; y = 1  [Xmin,Ymin]]
            #print("xymax")
            #print(xymax)
            #print("xymin")
            #print(xymin)

            xr = xymax[0]-xymin[0]
            yr = xymax[1]-xymin[1]
            #print("xr : {}\nyr : {}".format(xr,yr))
            xratio = 0.2
            yratio = 0.4            
            '''
            tbbox : LB,LT,RT,RB 
            [[Xmin , Xmin , Xmax , Xmax ]
             [Ymin , Ymax , Ymax , Ymin ]]
            '''
            tbbox = np.array(([[xymin[0] - (xr*xratio) , xymin[0] - (xr*xratio) , xymax[0] + (xr*xratio), xymax[0] + (xr*xratio)], \
                               [xymin[1] - (yr*yratio) , xymax[1] + (yr*yratio) , xymax[1] + (yr*yratio), xymin[1] - (yr*yratio)]]))
            ## 3D : scatter 2D line : plot
            # ax1.scatter(tbbox[0,:],tbbox[1,:],c="b",marker="o",label="tbbox")
            ##

            #print("After vstack")
            #print(tbbox)
            rota_angle = -1. * rota_angle
            rota_Matrix = np.array(([np.cos(rota_angle),-1.*np.sin(rota_angle)],[np.sin(rota_angle),np.cos(rota_angle)]))            
            tbbox = np.dot(rota_Matrix,tbbox)
            #print("tbbox")
            #print(tbbox)
            ##
            # ax0.scatter(tbbox[0,:],tbbox[1,:],c="g",marker="^",label="Processed bbox")
            ##            
            # ax0.legend(loc=9)
            # ax1.legend(loc=9)
            # if ShowPlot:
            #     plt.show()
            ##
            return tbbox            
            
        #elif :
        pass
    
    def Find_Vertex(self,min_angle_in_deg=70.):
        '''Find the turning vertex of the given shape.
        Input : a list of points in format [ [x1,y1,z1] , [x2,y2,z2] , ... ]
        Output : the index of the point in the shape.
        Complexity
        v0.1 @ 2016.07.18
        '''
        vtx = [] #for result
        for i in range(self.number_of_points-2):
            a=self.shape_np_pts[i+1]-self.shape_np_pts[i]
            a_n=npli.norm(a)
            b=self.shape_np_pts[i+2]-self.shape_np_pts[i+1]
            b_n=npli.norm(b)
            
            angle=np.arccos((np.dot(a,b)/a_n*b_n))
            if np.degrees(angle) >  min_angle_in_deg :
                vtx.append(i)
                
        vtx.append(self.number_of_points-1)
        return vtx
    
    def Regression_1D (self):
        ''' Find the coefficients (a,b,c) in formula Z= a*x + b*y + c.
        Input : None
        Output : (tuple) coefficient of formula (a,b,c) , (list) r-square 
        ----- Numpy Reference -----
        http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html
        '''
        A = np.vstack((self._get_x_arr(),self._get_y_arr(),np.ones(self.number_of_points))).T
        p,residuals,rank,s = npli.lstsq(A , self._get_z_arr(),rcond=-1) #rcond=-1 add on 20181212 to maintain the old behavior, note that r2 does not mean 
        # 20190611 Try to calculate r2
        r2 = 1. - residuals / (self._get_z_arr().size*self._get_z_arr().var()) #Return an array
        try:
            r2 = r2.tolist()[0]
        except IndexError as e:
            r2 = np.nan
        
        try:
            residuals = residuals.tolist()[0]
        except IndexError as e:
            residuals = np.nan
        finally:
            return p,residuals,rank,s,r2
        
    def Get_Strike(self):
        '''Return  the amuzith of strike
        Input : None (from regressioned formula 'a' and 'b' )
        Output : (float) amuzith of strike
        Diagram:
             90
           NE | SE      
        0 <---+--- 180
           NW | SW
             270
        You can get dd by add 90 degree to strike
        '''
        ang1=self._vector_angle(np.array([self._a,self._b]),np.array([-1,0]))
        ang=np.degrees(ang1)
        if self._b < 0. :
            return 360.-ang
        else:
            return ang
        pass
    
    def Get_Dip_Degree(self):
        return np.degrees(self._vector_angle2(np.array([self._a,self._b,-1.]),np.array([0.,0.,-1.]) ) )
        
    def Get_Dip_Direction(self,StringMode=False):
        dd = self.strike + 90.
        if  dd > 360 : dd-=360.
        #dd-=360. if  dd > 360 else print("ok")
        if StringMode:
            if dd > 315. and dd <= 360. :
                return "N"
            elif dd>=0. and dd <45 :
                return "N"
            elif dd > 45. and dd < 135. :
                return "E"
            elif dd > 135. and dd < 225. :
                return "S"
            elif dd > 225. and dd < 315. :
                return "W"
            elif dd == 45. :
                return "NE"
            elif dd == 135. :
                return "SE"
            elif dd == 225. :
                return "SW"
            elif dd == 315. :
                return "NW"
        else:       
            return dd 
    def calZ(self,x,y):
        """
        Calculate the z value based on the regression surface equation z = ax+by+c
        20181220 New
        """
        if (not isinstance(x,int)) and (not isinstance(x,float)):
            print("x must be an int or float. Stop calZ")
            return
        if (not isinstance(y,int)) and (not isinstance(y,float)):
            print("y must be an int or float. Stop calZ")
            return
        return self._a*x + self._b*y + self._c

class SDC(object):
    """計算位態使用的工具箱。
    變數:
    FileGDB_Path:FileGDP 路徑
    Layer_Name:圖層名稱
    Output_Path:輸出的檔案路徑(比如shape檔或csv檔)
    _Initialized:是否執行過UpdateParameter來設定參數
    """
    def __init__(self,FileGDB_Path,Layer_Name,Output_Path,Output_Overwrite=False):
        #定義一些變數
        self.FileGDB_Path = FileGDB_Path
        self.Layer_Name = Layer_Name
        self.Output_Path = Output_Path
        self.Output_Overwrite = Output_Overwrite
        # 以下是計算位態的相關參數
        self.FCL = None # Feature Class List，讀取完檔案時生出來的特徵清單
        self.bd_list = None
        # 有錯誤
        self.has_error = False
        print("Parameters:\nself.FileGDB_Path = {}\nself.Layer_Name = {}\nself.Output_Path = {}\nself.Output_Overwrite = {}".format(FileGDB_Path,Layer_Name,Output_Path,Output_Overwrite))
        if HasArcpy:
            arcpy.AddMessage("Parameters:\nself.FileGDB_Path = {}\nself.Layer_Name = {}\nself.Output_Path = {}\nself.Output_Overwrite = {}".format(FileGDB_Path,Layer_Name,Output_Path,Output_Overwrite))
        
    def points_to_shp(self,Parr,Farr,dst_path,overwrite=False):
        """
        將輸入的點陣列轉變為Shapefile。
        Parr：座標陣列，格式：[ [x1,y1,z1],[x2,y2,z2],... ]
        Farr：屬性表的陣列，第一欄為定義，第二欄以後為資料，故總數為N+1。第一欄資料：「種類=名稱」
            {S:ogr.OFTString,R:ogr.OFTReal,I:ogr.OFTInteger}
            若是使用「S」（字串），則後面可以指定長度
            [[S32=Name,R=Lat,R=Lon],["Point1",24.1,121.2],...]
        dst_path：儲存的路徑。
        overwrite：覆寫已經存在的檔案。
        """
        # 設定屬性表編碼：
        os.environ['SHAPE_ENCODING'] = "utf-8"
        
        # 確認檔案是否存在
        if os.path.exists(dst_path):
            if not overwrite:
                raise RuntimeError("The file already exists and the 'overwrite' is set to False.")
                return
            else:
                os.remove(dst_path)

        # set up the shapefile driver
        driver = ogr.GetDriverByName("ESRI Shapefile")
        
        # create the data source
        data_source = driver.CreateDataSource(dst_path)
        
        # create the spatial reference,
        srs = osr.SpatialReference()
        
        """參考資料來源：http://gis.rchss.sinica.edu.tw/qgis/?p=2823
        台灣常用的 EPSG代碼
        TM2（TWD97，中央經線121度）(適用臺灣本島，目前政府使用) ＝> EPSG:3826
        TM2（TWD97，中央經線119度）(適用澎湖，目前政府使用) ＝> EPSG:3825
        TM2（TWD67，中央經線121度）(適用臺灣本島，早期政府使用) ＝> EPSG:3828
        TM2（TWD67，中央經線119度）(適用澎湖，早期政府使用) ＝> EPSG:3827
        
        WGS84經緯度（全球性資料，如：GPS） ＝> EPSG:4326
        Spherical Mercator（圖磚、WMTS，如：Google Map） ＝> EPSG:3857
        
        TWD67經緯度（部分地籍圖圖解數化成果）＝> EPSG:3821
        TWD97經緯度（國土測繪中心發佈全國性資料）＝> EPSG:3824
        虎子山經緯度（日治時期陸測地形圖） ＝> EPSG:4236
        虎子山UTM zone 51N（中美合作軍用地形圖） ＝> EPSG:3829
        """
        
        srs.ImportFromEPSG(3826)
        
        # create the layer
        layer = data_source.CreateLayer("ASC_Result", srs, ogr.wkbPoint25D)
        
        # 建立屬性表
        # Create Fields based on the Farr[0] definition
        pat = re.compile(r"(?P<TypeCode>[IRS])(?P<StringLength>\d*)=(?P<FieldName>\w+)") # 指紋：「種類(長度)=名稱」
        FieldNameList = []
        FieldTypeList = []
        print("Fields in attribute table: \nName\tType\tLength(string)")
        for fn in  Farr[0]: # 讀取屬性表欄位定義並建立
            _tmp = re.search(pat,fn) # 解析欄位種類與名稱
            if isinstance(_tmp,type(None)): # 如果沒有 Match 到 指令
                print("The Field Code is not recognized. Function return. 無法解析屬性欄位資料，執行終止。")
                return
            # 印出欄位資訊：
            print("{}\t{}\t{}".format(_tmp.group("FieldName"),_tmp.group("TypeCode"),_tmp.group("StringLength")))
            
            # 屬性名稱與種類建立以後可以給後面使用
            FieldTypeList.append(_tmp.group("TypeCode"))
            FieldNameList.append(_tmp.group("FieldName"))
            
            # 解析欄位名稱與種類資料
            if _tmp.group("TypeCode")=="S":
                #字串資料需要指定寬度
                _FDef = ogr.FieldDefn(_tmp.group("FieldName"), ogr.OFTString)
                _FDef.SetWidth(int(_tmp.group("StringLength"))) # 設定字串欄位的長度
            elif _tmp.group("TypeCode")=="R":
                # Real Number
                _FDef = ogr.FieldDefn(_tmp.group("FieldName"), ogr.OFTReal)
            elif _tmp.group("TypeCode")=="I": 
                # Interger
                _FDef = ogr.FieldDefn(_tmp.group("FieldName"), ogr.OFTInteger)        
            else:
                print("The type code is not recognized. Function return. Type碼無法辨識（IRS其一），執行終止。")
                return
            
            layer.CreateField(_FDef)
        
        # 讀點寫入檔案與寫入屬性
        _f_num = len(FieldNameList) #有個欄位
        for ptid in range(len(Parr)):
            # 建立新的 Feature：
            _feature = ogr.Feature(layer.GetLayerDefn())
            #寫入屬性資料：
            for f_id in range(_f_num): #屬性資料
                _feature.SetField( FieldNameList[f_id] , Farr[ptid+1][f_id] ) #並未考慮欄位屬性直接寫入文字
            # 建立Geometry
            _wkt = "POINT({0[0]} {0[1]} {0[2]})".format(Parr[ptid]) # 建立點的文字述句
            _point = ogr.CreateGeometryFromWkt(_wkt) # 從述句建立點物件
            _feature.SetGeometry(_point) # 在 feature 中加入 geometry （點資料）
            
            layer.CreateFeature(_feature)
            _feature = None
        
        # 處理完畢以後存檔
        data_source = None
        
        return         
        
    def bd_list_to_shp(self,bdlist,dst,overwrite=False):
        """
        目的：
            將Bedding 物件清單轉為一個
        輸入：
            bdlist(List)：一個包含 bd 物件的清單，裡面有回歸的各項參數。
            dst(str)：存檔的路徑，必須以shp結尾，否則將自動加上。
            overwrite(bool)：是否覆寫已經存在的檔案。
        輸出：
            不會傳回任何物件，直接將結果輸出shape檔
        註解：
            座標：以幾何中心座標為主，包含在回歸面上的Z值
            輸出屬性表：
                CenX         ：幾何中心座標
                CenY         ：幾何中心座標
                CenZ         ：幾何中心座標
                Strike(R)    ：走向（RHR）
                Dip(R)       ：傾角
                pt_num(I)    ：點的數量
                a(R)         ：回歸面係數
                b(R)         ：回歸面係數
                c(R)         ：回歸面係數
                resident(R)  ：點與面的殘差和
                rank(R)      ：Rank of matrix a.
                R2(R)        ：決定係數
            回歸面的參數請見numpy說明：
            https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.linalg.lstsq.html
            相關係數目前並未計算。

        """
        # 確定結尾有 shp
        if not dst.endswith(".shp"):
            dst = dst + ".shp"
        # 準備容器來裝
        Parr = [] # 裝幾何中心座標 XYZ
        Farr = [] # 裝屬性表 
        # 準備 Farr
        # 第一欄資料為欄位定義
        Farr.append(["R=CenX","R=CenY","R=CenZ","R=Strike","R=Dip","I=pt_num","R=R2","R=a","R=b","R=c","R=residuals","R=rank","S15=Lib_ver"])
        # 準備每一點資料
        for bd in bdlist:
            Parr.append([bd.cenX,bd.cenY,bd.cenZ])
            Farr.append([str(bd.cenX),\
                         str(bd.cenY),\
                         str(bd.cenZ),\
                         str(bd.strike),\
                         str(bd.dip),\
                         str(bd.number_of_points),\
                         str(bd.r2),\
                         str(bd._a),\
                         str(bd._b),\
                         str(bd._c),\
                         str(bd.r),\
                         str(bd.rank),\
                         str(bd._libver)])
        # 呼叫 points_to_shp 來存檔
        self.points_to_shp(Parr,Farr,dst,overwrite)
    
    def bd_list_to_csv(self):
        # 若有錯誤結束執行
        if self.has_error:
            print("Function 'bd_list_to_csv' terminate due to the error occured in previous steps.")
            return
        # 確定結尾有 csv
        if not self.Output_Path.endswith(".csv"):
            self.Output_Path = self.Output_Path + ".csv"
        #定義欄位標題
        fieldnames = ["Centroid X","Centroid Y","Centroid Z","Strike","Dip","Number of points","R2","a","b","c","r","rank","libver","Create Time"]
        #開始寫入資料
        if os.path.exists(self.Output_Path) == False:
            #檔案不存在，建立新的檔案
            print("Creating new CSV file:\n{}".format(self.Output_Path))
            with io.open(self.Output_Path,"at",encoding="utf-8",newline='') as csvfile:
                #寫入檔頭(標題)       
                writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
                writer.writeheader()
                #寫入每筆計算資料
                for bd in self.bd_list:
                    writer.writerow({\
                    "Centroid X":str(round(bd.cenX,2)),\
                    "Centroid Y":str(round(bd.cenY,2)),\
                    "Centroid Z":str(round(bd.cenZ,2)),\
                    "Strike":str(bd.strike),\
                    "Dip":str(bd.dip),\
                    "Number of points":str(bd.number_of_points),\
                    "R2":str(bd.r2),\
                    "a":str(bd._a),\
                    "b":str(bd._b),\
                    "c":str(bd._c),\
                    "r":str(bd.r),\
                    "rank":str(bd.rank),\
                    "libver":str(bd._libver),
                    "Create Time":time.strftime("%Y-%m-%d %H:%M:%S")\
                    })
            print(r"Please visit {} to check the csv file.".format(self.Output_Path))
        elif os.path.exists(self.Output_Path) == True:
            with io.open(self.Output_Path,"at",encoding="utf-8",newline='') as csvfile:
                writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
                for bd in self.bd_list:
                    writer.writerow({\
                    "Centroid X":str(bd.cenX),\
                    "Centroid Y":str(bd.cenY),\
                    "Centroid Z":str(bd.cenZ),\
                    "Strike":str(bd.strike),\
                    "Dip":str(bd.dip),\
                    "Number of points":str(bd.number_of_points),\
                    "a":str(bd._a),\
                    "b":str(bd._b),\
                    "c":str(bd._c),\
                    "r":str(bd.r),\
                    "rank":str(bd.rank),\
                    "libver":str(bd._libver),
                    "Create Time":time.strftime("%Y-%m-%d %H:%M:%S")\
                    })
            print(r"Please visit {} to check the csv file.".format(self.Output_Path))
        print("[Done] bd_list_to_csv")
        return
    
    def FileGDB_to_BDL(self):
        ### 使用「FileGDB」或是「OpenFileGDB」去讀取 FileGDB 資料庫。###
        driver = ogr.GetDriverByName("FileGDB")
        if isinstance(driver,NoneType):
            print("The driver 'FileGDB' is NOT available in this system!")
            if HasArcpy:
                arcpy.AddMessage("The driver 'FileGDB' is NOT available in this system!")
        # Try to load FileGDB using "OpenFileGDB" driver.
        driver = ogr.GetDriverByName("OpenFileGDB")
        if isinstance(driver,NoneType):
            print("The driver 'OpenFileGDB' is NOT available in this system!")
            print("Please confirm the driver installation needed for accessing FileGDB")
            if HasArcpy:
                arcpy.AddMessage("The driver 'OpenFileGDB' is NOT available in this system!")
                arcpy.AddMessage("Please confirm the driver installation needed for accessing FileGDB")
        else:
            print("Using 'OpenFileGDB' as driver")
            if HasArcpy:
                arcpy.AddMessage("Using 'OpenFileGDB' as driver")
        # 開啟GDB檔案
        print("Opening the FileGDB")
        if HasArcpy:
            arcpy.AddMessage("Opening the FileGDB")
        try:
            gdb = driver.Open(self.FileGDB_Path,0) # 0 means read-only
        except Exception as e:
            print(e)
            if HasArcpy:
                arcpy.AddMessage(e)
        # 測試是否成功開啟：
        if isinstance(gdb,NoneType):
            print("Fail to open the geodatabase.")
            if HasArcpy:
                arcpy.AddMessage("Fail to open the geodatabase.")
            sys.exit(1)
        else:
            print("Opened the geodatabase")
            if HasArcpy:
                arcpy.AddMessage("Opened the geodatabase")
        ### 開啟特定圖層 ###
        # 讀取資料庫中特定的圖層（此例是lineation，可用小工具「get_gdb_layers」取得所有名稱）
        self.FCL = gdb.GetLayerByName(self.Layer_Name)
        
        ### 執行位態回歸 ###
        # 筆記：create_bedding_list
        Ts = time.time() #設定一個小計時器讓接下來如果變成無限回圈可以中斷    
        Cnt = 0 # 計數用的變數
        self.bd_list = []
        # 開始迭代
        Fe = self.FCL.GetNextFeature() # 讀取第一個 Feature
        print("Creating bedding list...")
        while Fe:
            # 測試時為了避免無限迴圈而設計的炸彈
            #if time.time() - Ts > 20: #20秒後自動中斷
            #    break
            #這邊是針對每個FC要做的處理：
            G = Fe.GetGeometryRef()
            GG = G.GetGeometryRef(0)
            #print("GeometryType \t {}\nGeometryName \t {}".format(GG.GetGeometryType(),GG.GetGeometryName()))
            #GGp = GG.GetPoints()
            #建立Bedding物件
            bd = bedding(GG.GetPoints())
            #取得幾何中心座標，並根據回歸面計算出z值
            bd.cenX = GG.Centroid().GetX()
            bd.cenY = GG.Centroid().GetY()
            bd.cenZ = bd.calZ(GG.Centroid().GetX(),GG.Centroid().GetY()) # 使用 ogr 模組取得幾何中心，並用 bedding 物件計算出在回歸面上的位置
            # 將結果放入結果清單中
            self.bd_list.append(bd)
            # 關閉 Feature，釋放資源
            Fe.Destroy() #　關閉處理完的 FC，可是官網說是為了相容舊的版本？？
            Fe  = self.FCL.GetNextFeature()
            
            Cnt = Cnt + 1
            sys.stdout.write("*")
        sys.stdout.write("\nDone!\n")
        self.FCL.ResetReading() # 重設讀取指標的位移，讓他可以再次被迭代
        print("Processed {} features".format(Cnt))
        if HasArcpy:
            arcpy.AddMessage("Processed {} features".format(Cnt))
        return
        
    def SDC_FileGDB_to_shp(self):
        self.FileGDB_to_BDL()
        ### 輸出 ### 
        # 輸出檔案到 Shapefile
        # 筆記：bd_list_to_shp
        print("Saving shapefile...")
        self.bd_list_to_shp(self.bd_list,self.Output_Path,True)
        print("Saving csv...")
        self.bd_list_to_csv()
        return
        
class SDC2shp(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "SDC2shp-20190611"
        self.description = "Strike and dip calculator by CWsun using GDAL."
        self.canRunInBackground = True    
        try:
            from osgeo import gdal
            self.GDAL_installed = True
            print("Successfully import pygdal")
            if HasArcpy:
                arcpy.AddMessage("Successfully import pygdal")
        except ImportError as e:
            self.GDAL_installed = False
            print("No pygdal is found under current python runtime environment.")
            if HasArcpy:
                arcpy.AddMessage("No pygdal is found under current python runtime environment.")
        return

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(\
            displayName="Input FileGDB feature class",\
            name="InFC",\
            datatype="DEFeatureClass",\
            parameterType="Required",\
            direction="Input")
        param1 = arcpy.Parameter(\
            displayName="Output shapefile path",\
            name="OutPath",\
            datatype="DEFeatureClass",\
            parameterType="Required",\
            direction="None")
        param2 = arcpy.Parameter(\
            displayName="Message",\
            name="Message",\
            datatype="GPString",\
            parameterType="Optional",\
            direction="None",\
            enabled=False)
        param1.parameterDependencies = [param0.name]
        params = [param0,param1,param2]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute.
        使用是否已安裝 GDAL 來確認可否執行"""
        return self.GDAL_installed

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        if parameters[0].altered:
            if not isinstance(parameters[0],type(None)):
                parameters[1].value = oph.join(oph.dirname(oph.dirname(parameters[0].valueAsText)),"SDC_"+oph.basename(parameters[0].valueAsText)+time.strftime("%Y%m%d-%H%M")+".shp")
            else:
                parameters[0].setErrorMessage("Please specify the input feature class.")
                parameters[1].value = ""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        InFC = parameters[0].valueAsText
        InFGDB = oph.dirname(InFC)
        InLayer = oph.basename(InFC)
        OutShp = parameters[1].valueAsText
        print("InFGDB:{}\nInLayer:{}\nOutShp:{}".format(InFGDB,InLayer,OutShp))
        if HasArcpy:
            arcpy.AddMessage("InFGDB:{}\nInLayer:{}\nOutShp:{}".format(InFGDB,InLayer,OutShp))
        #RUN
        Tool=SDC(InFGDB,InLayer,OutShp,True)
        Tool.SDC_FileGDB_to_shp()
        return
        
# if __name__ == "__main__":
#     arg = sys.argv
#     if not (len(arg) == 2): # One parameters
#         print("[Error] Only one parameter is acceptable.")
#         sys.exit()
#     InPath = arg[1]
#     InFGDB = oph.dirname(InPath)
#     InLayer = oph.basename(InPath)
#     OutShp = oph.join(oph.dirname(oph.dirname(InPath)),"SDC_"+oph.basename(InPath)+time.strftime("%Y%m%d-%H%M")+"CLI.shp")
#     print("InFGDB:{}\nInLayer:{}\nOutShp:{}".format(InFGDB,InLayer,OutShp))
#     Tool=SDC(InFGDB,InLayer,OutShp,True)
#     Tool.SDC_FileGDB_to_shp()
    
"""
Change log:
20190611 CW first release.
20190826 CW add command line usage
20200812 CW Modified the code for pure GDAL environment
20221129 CW Comment some drawing code and make it run in AGP
"""