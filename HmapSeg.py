from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
from skimage.morphology import disk, square
from skimage.filters.rank import median
from skimage import filters
from skimage.morphology import erosion, dilation, opening, closing, white_tophat,binary_opening,thin
from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
from shapely.ops import polygonize
import pprint
import configparser
import sys
import os.path
#llcrnrlon=22, llcrnrlat=30,urcrnrlon=30, urcrnrlat=42


class HmapSeg():

  def __init__(self):
      pass

  def drawWorldMap(self):
      map = Basemap()
      map.drawcoastlines()
      map.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0])
      map.drawmeridians(np.arange(-180,181,30), labels=[0,0,0,1])
      plt.show()

  def convertMatToGray(self,matA):
      max_v = np.max(matA)
      fac = 255.0/max_v
      matA = np.round(matA*fac).astype(np.uint8)
      return matA

  def create_Dictionary_NARI(self,file_name="nari_dynamic.csv",file_save = "dict_nari",v=1000000):
      nari = {}
      #counter = 0
      #sourcemmsi,navigationalstatus,rateofturn,speedoverground,courseoverground,trueheading,lon,lat,t
      counter = 0
      f_count = 0
      with open(file_name) as infile:
           for line in infile:

               if counter == 0:
                  counter = counter + 1
                  continue

               line_split = line.split(",")

               if line_split[0] not in nari.keys():

                  temp_var = np.zeros((1,4))
                  temp_var[0,0] = float(line_split[8]) #time
                  temp_var[0,1] = float(line_split[6]) #lon
                  temp_var[0,2] = float(line_split[7]) #lat
                  temp_var[0,3] = float(line_split[3]) #speed

                  nari[line_split[0]] = temp_var
               else:

                  temp_var = np.zeros((1,4))
                  temp_var[0,0] = float(line_split[8]) #time
                  temp_var[0,1] = float(line_split[6]) #lon
                  temp_var[0,2] = float(line_split[7]) #lat
                  temp_var[0,3] = float(line_split[3]) #speed
                  nari[line_split[0]] = np.vstack((nari[line_split[0]],temp_var))
  
               
               if counter%v == 0:
                  joblib.dump(nari, file_save+"_"+str(f_count)+".sav")
                  nari = {}
                  f_count = f_count + 1
                  print(counter)
               counter = counter + 1

  def plot_play(self,file_name="dict_nari_5.sav"):
      dict_var = joblib.load(file_name)
      import math
      print(len(dict_var.keys()))      
      m_vec = np.array([])
      std_vec = np.array([])
      mean_vec = np.array([])
      mask_vessel = np.ones((len(dict_var.keys()),),dtype=int)
      counter = 0
      for vessel in dict_var.keys():
          temp = dict_var[vessel]
          
          t = temp[:,0]
          idx = np.argsort(t)
          temp = temp[idx,:]
          x1 = temp[0:-1,1]
          x2 = temp[1:,1] 
           
          y1 = temp[0:-1,2]
          y2 = temp[1:,2]

          if len(x1) <> 0:
             m = np.zeros(x1.shape)

             #print(m)
             #print("x1 = "+str(x1))
          
             for k in range(len(x1)):
                 if not np.allclose((x2[k]-x1[k]),0):
                    m[k] = (y2[k]-y1[k])/(x2[k]-x1[k])
                 #print(m)   
             m_vec = np.append(m_vec,m)
         
             #print(m)   
             std_vec = np.append(std_vec,np.std(m))
             mean_vec = np.append(std_vec,np.mean(m))
             if np.std(m) > 1:
                mask_vessel[counter] = 0   
             #print(m)   

             #if math.isnan(np.std(m)):
             #   print("m_if = "+str(m))
             #   break 
             if np.std(m) > 1:
                #plt.plot(temp[:-1,1],m)
                plt.plot(temp[:,1],temp[:,2])
                #plt.ylim(-5,5)
          else:
             mask_vessel[counter] = 0
          counter = counter+1 

      plt.show(mean_vec,std_vec,'rx')
      plt.show() 
      m_vec_zero_free = np.array([])

      for k in range(len(m_vec)):
          if not np.allclose(m_vec[k],0):
             m_vec_zero_free = np.append(m_vec_zero_free,m_vec[k])

      plt.semilogy()
           
      m_vec_zero_free =m_vec_zero_free[m_vec_zero_free<20]
      m_vec_zero_free = m_vec_zero_free[m_vec_zero_free>-20]
      n, bins, patches = plt.hist(x=m_vec_zero_free,bins=500,color='#0504aa',alpha=0.7)
      plt.grid(axis='y', alpha=0.75)
      plt.xlabel('Value')
      plt.ylabel('Frequency')
      plt.title('My Very Own Histogram')
      #plt.text(23, 45, r'$\mu=15, b=3$')
      plt.show()
      plt.plot(std_vec)
      print(std_vec)
      plt.show()
      n, bins, patches = plt.hist(x=std_vec,bins=100,color='#0504aa',alpha=0.7)
      plt.grid(axis='y', alpha=0.75)
      plt.xlabel('Value')
      plt.ylabel('Frequency')
      plt.title('My Very Own Histogram')
      #plt.text(23, 45, r'$\mu=15, b=3$')
      plt.show()

      #print(t[idx])
      #break
  def gridData_to_2DMap_NARI(self,file_name="nari_dynamic.csv",N=10001,v=1000000,file_save='TwoDGridNARI.sav'):
      x = np.linspace(-180,180,N,endpoint=True)
      x_value = x + (x[1]-x[2])/2.0
      x_value = x_value[:-1]

      y = np.linspace(-90,90,N,endpoint=True)
      y_value = y + (y[1]-y[2])/2.0
      y_value = y_value[:-1]

      matriks = np.zeros((N-1,N-1),dtype=int)

      counter = 0

      with open(file_name) as infile:
           for line in infile:
               #print(line)
               
               if counter == 0:
                  counter = counter + 1
                  continue
               
               line_split = line.split(",")

               lon = float(line_split[6]) 
               lat = float(line_split[7])

               index_x = np.searchsorted(x,lon)
               index_y = np.searchsorted(y,lat)

               if (index_x == N):
                  index_x = index_x - 2
               elif (index_x > 0):
                  index_x = index_x - 1
           
               if (index_y == N):
                  index_y = index_y - 2
               elif (index_y > 0):
                  index_y = index_y - 1
 
               matriks[index_y,index_x]+=1

               counter = counter + 1
               if (counter % v == 0):
                  print("counter = ",counter) 


      joblib.dump(matriks, file_save)  

  def gridData_to_2DMap(self,file_name="8monthsAIS.txt",N=10001,v=1000000,file_save='TwoDGrid.sav'):
      x = np.linspace(-180,180,N,endpoint=True)
      x_value = x + (x[1]-x[2])/2.0
      x_value = x_value[:-1]

      y = np.linspace(-90,90,N,endpoint=True)
      y_value = y + (y[1]-y[2])/2.0
      y_value = y_value[:-1]

      matriks = np.zeros((N-1,N-1),dtype=int)

      counter = 0

      with open(file_name) as infile:
           for line in infile:
               
               line_split = line.split(" ")

               lat = float(line_split[2]) 
               lon = float(line_split[3])

               index_x = np.searchsorted(x,lon)
               index_y = np.searchsorted(y,lat)

               if (index_x == N):
                  index_x = index_x - 2
               elif (index_x > 0):
                  index_x = index_x - 1
           
               if (index_y == N):
                  index_y = index_y - 2
               elif (index_y > 0):
                  index_y = index_y - 1
 
               matriks[index_y,index_x]+=1

               counter = counter + 1
               if (counter % v == 0):
                  print("counter = ",counter) 


      joblib.dump(matriks, file_save)  

  def plotGriddedData(self,file_save='TwoDGrid.sav',vmax=50,cmv='Reds',image_name='test.pdf',N=10001):
      map = Basemap()
      map.drawcoastlines()
      matriks = joblib.load(file_save)
      map.imshow(matriks,vmax=vmax,cmap=plt.cm.get_cmap(cmv))
      plt.savefig(image_name)
      plt.show()

  def polygonSegmentation(self,file_save='TwoDGrid.sav',cmv='hot',N=10001, mask_file="mask.sav", water_mask = "water_mask.sav", resolution='h',llcrnrlon=0, llcrnrlat=35,urcrnrlon=15, urcrnrlat=45, m_size = 5, threshold=150, o_size = 3, min_distance=18, min_angle=20, h_threshold=0.6,num_peaks=10,config_file="",save_fig=True):
      dir_name = config_file[:-4]
      
      if not os.path.isdir(dir_name):
         #print("Hallo")
         os.system("mkdir "+ dir_name)
      
      #PLOT MAP
      print("EXTRACTING SUBMAP")
      map = Basemap(resolution=resolution,llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
      map.drawcoastlines()
      parallels = np.linspace(llcrnrlat,urcrnrlat,5)
      map.drawparallels(parallels,labels=[1,0,0,0])
      meridians = np.linspace(llcrnrlon,urcrnrlon,5)
      map.drawmeridians(meridians,labels=[0,0,0,1])

      #EXTRACT SUB HEATMAP ACCORDING TO COORDINATES GIVEN 
      x = np.linspace(-180,180,N,endpoint=True)
      dx = np.absolute(x[1]-x[2])
      x_value = x + (x[1]-x[2])/2.0
      x_value = x_value[:-1]

      y = np.linspace(-90,90,N,endpoint=True)
      dy = np.absolute(y[1]-y[2])
      y_value = y + (y[1]-y[2])/2.0
      y_value = y_value[:-1]

      index_x_1 = np.searchsorted(x,llcrnrlon)-1
      index_x_2 = np.searchsorted(x,urcrnrlon)-1

      index_y_1 = np.searchsorted(y,llcrnrlat)-1
      index_y_2 = np.searchsorted(y,urcrnrlat)-1

      cost_line_mask = joblib.load(mask_file)
      cost_line_mask = np.absolute(cost_line_mask-1)
      
      matriks = joblib.load(file_save)
      sub_m = matriks[index_y_1:index_y_2,index_x_1:index_x_2]
      
    
      #EXTRACT_WATER_MASK
      water_mask = joblib.load(water_mask)  
      
      #TAKE LOG
      sub_m = np.log(sub_m+1)
      sub_m_rev = sub_m[::-1,:] #MAKE COPY OF ORIGINAL WILL USE DURING SEGEMENTATION STAGE      

      #PLOT SUB-HEATMAP      
      cs = map.imshow(sub_m,vmax=8)
      sub_m = self.convertMatToGray(sub_m)  
      cbar = map.colorbar(cs,location='bottom',pad="5%")
      cbar.set_label('log(#)')
      if save_fig:
         plt.savefig('./'+dir_name+'/'+dir_name+".png")
      else:
         plt.show()
      plt.close()
      
      print("MEDIAN FILTER")
      #MEDIAN FILTER DATA AND BINARY THRESHOLD
      med = sub_m*cost_line_mask #remove coastline data
      med = median(med, disk(m_size)) #media filter data
      med = sub_m-med #highlight linear tracks --- similar to rfi detection
      
      med = med<threshold #thresholding

      med[water_mask==0] = np.amin(med)
      med[cost_line_mask==0] = np.amin(med)
      med[sub_m==0] = 0

      map.drawcoastlines()
      map.imshow(med)
      
      if save_fig:
         plt.savefig('./'+dir_name+'/'+dir_name+"_binary.png")
      else:
         plt.show()
      plt.close()
       
      #APPLY MORPHOLOGICAL OPENING
      print("OPENING")
      selem = square(o_size) #opening
      opened = opening(med,selem)
      
      map.drawcoastlines()
      map.imshow(opened)
      
      if save_fig:
         plt.savefig('./'+dir_name+'/'+dir_name+"_opened.png")
      else:
         plt.show()
      plt.close()
      
      #HOUGH TRANSFORM
      #################################
      print("HOUGH TRANSFORM")
      sub_m_copy = opened[::-1,:]
      h, theta, d = hough_line(sub_m_copy)
      
      plt.imshow(np.log(1 + h),extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]])#aspect=1/1.5
      plt.axes().set_aspect('auto', adjustable='box')
      if save_fig:
         plt.savefig('./'+dir_name+'/'+dir_name+"_ht.png")
      else:
         plt.show()
      plt.close()
    
      S_land = []
      for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=min_distance, min_angle=min_angle, threshold=h_threshold*np.max(h),num_peaks=num_peaks)):#numpeaks
          y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)

          y1 = (dist - sub_m_copy.shape[1] * np.cos(angle)) / np.sin(angle)

          S_land.append(((0,y0), (sub_m_copy.shape[1], y1)))
          plt.plot((0, sub_m_copy.shape[1]), (y0, y1), '-r')
      
      plt.imshow(sub_m_copy)
      if save_fig:
         plt.savefig('./'+dir_name+'/'+dir_name+"_ht_lines.png")
      else:
         plt.show()
      plt.close() 
      #################################

      #FIND POLYGONS USING SHAPELY
      #################################
      print("POLYGONIZATION")
      #https://gis.stackexchange.com/questions/58245/generate-polygons-from-a-set-of-intersecting-lines
      #lines ==> MultiLineString {:: M}
      #add a tiny buffer, say eps {:: MB}
      #region ==> Polygon {:: P} (region here is a square)
      #P.difference(MB) {resulting polygons}

      from shapely.geometry import MultiLineString
      from shapely.geometry import Polygon
      lines = MultiLineString(S_land)
      linesb = lines.buffer(0.1)
      P = Polygon([(0,0),(0,sub_m_copy.shape[0]),(sub_m_copy.shape[1],sub_m_copy.shape[0]),(sub_m_copy.shape[1],0),(0,0)])
      p_new = P.difference(linesb)
      for p in p_new:
          c = np.array(p.exterior.coords)
          #print(c.shape)
          #print(c)
          #plt.plot(c[:,0],c[:,1])
      #plt.show()
      #################################

      #GRID POLYGONS TO 2D GRID
      #################################
      from matplotlib.path import Path
      poly_mask = np.zeros(sub_m_copy.shape)
      counter = 1
      for p in p_new:
          poly_path=Path(p.exterior.coords)
          
          
          nx, ny = sub_m_copy.shape[1], sub_m_copy.shape[0]
          
          # Create vertex coordinates for each grid cell...
          # (<0,0> is at the top left of the grid in this system)
          x, y = np.meshgrid(np.arange(nx), np.arange(ny))
          x, y = x.flatten(), y.flatten()

          points = np.vstack((x,y)).T
          grid = poly_path.contains_points(points)
          mask = grid.reshape((ny,nx))

          #plt.imshow(mask)
          #plt.show() sub_m_rev = sub_m[::-1,:]
          #print(counter)
          poly_mask[mask] = counter
          counter += 1
      #################################
      
      #for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=min_distance, min_angle=min_angle, threshold=h_threshold*np.max(h),num_peaks=num_peaks)):#numpeaks
      #    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
      #    print("y0 = "+str(y0))

      #    y1 = (dist - sub_m_copy.shape[1] * np.cos(angle)) / np.sin(angle)
      #    print("y1 = "+str(y1))
      #    S_land.append(((0,y0), (sub_m_copy.shape[1], y1)))
      #    plt.plot((0, sub_m_copy.shape[1]), (y0, y1), '-r')   
      
      #OVERLAY COASTLINE AND WATERMASK ON POLYGON MASK
      #################################
      water_mask = water_mask[::-1,:]
      cost_line_mask = cost_line_mask[::-1,:]
      poly_mask[water_mask==0] = -2
      poly_mask[cost_line_mask==0] += 1 
      cost_line_mask[poly_mask==-1] = 1
      poly_mask[poly_mask == -1] = -2
      poly_mask[cost_line_mask==0] = -1
      #################################      

      #SEGMENT MAP BASED ON POLYGON MASK OLD
      #################################
      #segmentation_map = np.zeros(poly_mask.shape)

      #for k in range(counter):
      #    segmentation_map[poly_mask==k] = np.average(sub_m_rev[poly_mask==k])
      

      #segmentation_map[poly_mask==-1] = np.average(sub_m_rev[poly_mask==-1])
      #segmentation_map[poly_mask==-2] = 0

      #for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=min_distance, min_angle=min_angle, threshold=h_threshold*np.max(h),num_peaks=num_peaks)):#numpeaks
      #    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
      #    print("y0 = "+str(y0))

      #    y1 = (dist - sub_m_copy.shape[1] * np.cos(angle)) / np.sin(angle)
      #    print("y1 = "+str(y1))
      #    S_land.append(((0,y0), (sub_m_copy.shape[1], y1)))
      #    plt.plot((0, sub_m_copy.shape[1]), (y0, y1), '-r')   
       
      #plt.imshow(segmentation_map)
      
      #plt.show()
      #################################
      
      #SEGMENT MAP BASED ON POLYGON MASK
      #################################
      print("SEGMENTATION")
      map.drawcoastlines()
      segmentation_map = np.zeros(poly_mask.shape)

      for k in range(counter):
          segmentation_map[poly_mask==k] = np.average(sub_m_rev[poly_mask==k])
      

      segmentation_map[poly_mask==-1] = np.average(sub_m_rev[poly_mask==-1])
      segmentation_map[poly_mask==-2] = 0
      
      S = []
      for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=min_distance, min_angle=min_angle, threshold=h_threshold*np.max(h),num_peaks=num_peaks)):#numpeaks
          
          y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
          y1 = (dist - sub_m_copy.shape[1] * np.cos(angle)) / np.sin(angle)

          S.append(((llcrnrlon,urcrnrlat-1*dy*y0),(urcrnrlon,urcrnrlat*dy*y1)))
          #print("y1 = "+str(dy*y1))
          #print("y2 = "+str(dy*y0))
          map.plot((llcrnrlon, urcrnrlon), (urcrnrlat-1*dy*y0, urcrnrlat-1*dy*y1), '-r')
          #break
      cs = map.imshow(segmentation_map[::-1,:])
      #cs = map.imshow(sub_m)

      parallels = np.linspace(llcrnrlat,urcrnrlat,5)
      map.drawparallels(parallels,labels=[1,0,0,0])
      meridians = np.linspace(llcrnrlon,urcrnrlon,5)
      map.drawmeridians(meridians,labels=[0,0,0,1])
      cbar = map.colorbar(cs,location='bottom',pad="5%")
      cbar.set_label('log(#)')
      if save_fig:
         plt.savefig('./'+dir_name+'/'+dir_name+"_segmented.png")
      else:
         plt.show()
      plt.close() 

      #PLOT EXTRACTED TRACKS ON ORIGINAL IMAGE
      #################################
      map.drawcoastlines()
           
      S = []
      for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=min_distance, min_angle=min_angle, threshold=h_threshold*np.max(h),num_peaks=num_peaks)):#numpeaks
          
          y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
          y1 = (dist - sub_m_copy.shape[1] * np.cos(angle)) / np.sin(angle)

          S.append(((llcrnrlon,urcrnrlat-1*dy*y0),(urcrnrlon,urcrnrlat*dy*y1)))
          #print("y1 = "+str(dy*y1))
          #print("y2 = "+str(dy*y0))
          map.plot((llcrnrlon, urcrnrlon), (urcrnrlat-1*dy*y0, urcrnrlat-1*dy*y1), '-r')
          #break
      cs = map.imshow(sub_m_rev[::-1,:])

      parallels = np.linspace(llcrnrlat,urcrnrlat,5)
      map.drawparallels(parallels,labels=[1,0,0,0])
      meridians = np.linspace(llcrnrlon,urcrnrlon,5)
      map.drawmeridians(meridians,labels=[0,0,0,1])
      cbar = map.colorbar(cs,location='bottom',pad="5%")
      cbar.set_label('log(#)')
      if save_fig:
         plt.savefig('./'+dir_name+'/'+dir_name+"_lines.png")
      else:
         plt.show()
      plt.close() 
      #################################



  def applyKMeans(self,file_save='TwoDGrid.sav',cmv='hot',N=10001,mask_file="mask.sav"):
      map = Basemap(resolution='h',llcrnrlon=22, llcrnrlat=30,urcrnrlon=30, urcrnrlat=42)
      map.drawcoastlines()

      x = np.linspace(-180,180,N,endpoint=True)
      dx = np.absolute(x[1]-x[2])
      x_value = x + (x[1]-x[2])/2.0
      x_value = x_value[:-1]

      y = np.linspace(-90,90,N,endpoint=True)
      dy = np.absolute(y[1]-y[2])
      y_value = y + (y[1]-y[2])/2.0
      y_value = y_value[:-1]

      index_x_1 = np.searchsorted(x,22)-1
      index_x_2 = np.searchsorted(x,30)-1

      index_y_1 = np.searchsorted(y,30)-1
      index_y_2 = np.searchsorted(y,42)-1

      m = joblib.load(mask_file)
      m = np.absolute(m-1)
      matriks = joblib.load(file_save)
      sub_m = matriks[index_y_1:index_y_2,index_x_1:index_x_2]
      matriks=''
      print(sub_m.shape)
      
      sub_m = np.log(sub_m)
      sub_m = self.convertMatToGray(sub_m)
      sub_m = sub_m*m

      from skimage import data 
      from skimage.morphology import disk
      from skimage.filters.rank import median
      img = np.copy(sub_m)
      med = median(img, disk(2)) 
      #map.imshow(sub_m-med,vmax=5, vmin = 0)
      temp = sub_m-med
      #temp[temp<=0] = 0
      #temp[np.logicaltemp>0] = 1
      #map.imshow(temp)      
      #plt.show()
      #histo = plt.hist(np.absolute(sub_m.ravel()-med.ravel()), bins=np.arange(0, 256))
      #plt.show()
      temp[temp>30] = 0
      temp[temp<0] = 0
      #temp[temp<>0] = 1
      sub_m = np.copy(temp)
      
      
      map.drawcoastlines()
      old = np.copy(sub_m)
      h,theta,d = self.houghtransformbrightness(old[::-1,:])
      from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
      S_land = []
      for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=10, min_angle=10, threshold=0.5*np.max(h),num_peaks=5)):#numpeaks
          y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
          print("y0 = "+str(y0))

          y1 = (dist - sub_m.shape[1] * np.cos(angle)) / np.sin(angle)
          print("y1 = "+str(y1))
          S_land.append(((0,sub_m.shape[1]), (y0, y1)))
          plt.plot((0, sub_m.shape[1]), (y0, y1), '-r')
          #break
      plt.imshow(sub_m[::-1,:],cmap=plt.cm.get_cmap("jet"))
      #plt.axes().set_aspect('auto', adjustable='box')
      plt.show()
      plt.close() 
      
      #FILTER OR NOT
      #from skimage.morphology import disk
      #from skimage.filters.rank import median
      #sub_m = median(sub_m, disk(1))

      im = map.imshow(sub_m,cmap=plt.cm.get_cmap('gray'))
      plt.show()

      X = np.reshape(sub_m,(sub_m.shape[0]*sub_m.shape[1],1))


      X2 = X[X<>0]

      X2 = np.reshape(X2,(len(X2),1))

      
      from sklearn.cluster import KMeans
      
      kmeans = KMeans(n_clusters=3, random_state=0).fit(X2)
      
      #KMeans(n_clusters=2, random_state=0).fit(X)
      
      
      X[X<>0] = kmeans.labels_+1   

      sub_m_copy = np.reshape(X,(sub_m.shape[0],sub_m.shape[1]))
      sub_m_copy = temp
      map = Basemap(resolution='h',llcrnrlon=22, llcrnrlat=30,urcrnrlon=30, urcrnrlat=42)
      map.drawcoastlines()
      #threshold 
      #sub_copy
      #map.imshow(sub_m_copy)
      level = np.zeros(sub_m_copy.shape,dtype=sub_m_copy.dtype)
      #level[sub_m_copy==3] = 1
      
      from skimage.morphology import skeletonize
      # perform skeletonization
      #skeleton = skeletonize(level)
      #map.imshow(level)
      #plt.show()
 

      from matplotlib import cm
      from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
      sub_m_copy = sub_m_copy[::-1,:]
      # Classic straight-line Hough transform
      h, theta, d = hough_line(sub_m_copy)
      
      plt.imshow(np.log(1 + h),extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],cmap=cm.gray)#aspect=1/1.5
      plt.axes().set_aspect('auto', adjustable='box')
      plt.show()
      plt.close()
      
      S_land = []
      for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=20, min_angle=10, threshold=0.5*np.max(h),num_peaks=20)):#numpeaks
          y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
          print("y0 = "+str(y0))

          y1 = (dist - sub_m_copy.shape[1] * np.cos(angle)) / np.sin(angle)
          print("y1 = "+str(y1))
          S_land.append(((0,sub_m_copy.shape[1]), (y0, y1)))
          plt.plot((0, sub_m_copy.shape[1]), (y0, y1), '-r')
          #break
      plt.imshow(sub_m_copy,cmap=plt.cm.get_cmap(cmv))
      #plt.axes().set_aspect('auto', adjustable='box')
      plt.show()
      plt.close() 

      map.imshow(old)
      map.drawcoastlines()
      S = []
      for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=20, min_angle=10, threshold=0.5*np.max(h),num_peaks=7)):#numpeaks
          
          y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
          y1 = (dist - sub_m_copy.shape[1] * np.cos(angle)) / np.sin(angle)

          S.append(((22,42-1*dy*y0),(30,42-1*dy*y1)))
          #print("y1 = "+str(dy*y1))
          #print("y2 = "+str(dy*y0))
          map.plot((22, 30), (42-1*dy*y0, 42-1*dy*y1), '-r')
          #break
      #plt.show()
      
      from lsi import intersection

      # S is a list of tuples of the form: ((x,y), (x,y))
      #S = [((0,1),(2,1)),((0,2),(1,1))]
      
      i = intersection(S) 
      print(i.keys())
      
      print(i)

      for s in i.keys():
          map.plot(s[0],s[1],"rs")
      #plt.show()  

      S_int = self.intersectionsLand(S)
      for s in S_int:
          map.plot(s[0],s[1],"bs")
      #plt.show() 

      for s in S:
          S_new = self.divideIntoLineSegments(s,5000,llcrnrlon=22, llcrnrlat=30,urcrnrlon=30, urcrnrlat=42) 
          print "S_new = ",S_new 
          for s in S_new:
              map.plot((s[0][0],s[1][0]),(s[0][1],s[1][1]),"g")
              map.plot(s[0][0],s[0][1],"gs")
 
              map.plot(s[1][0],s[1][1],"ms")
 
      
      plt.show()

 
  def createMask(self,llcrnrlon=0, llcrnrlat=35,urcrnrlon=15, urcrnrlat=45,N_row=555+1,N_column=417+1,resolution='h',plot_img=False):
      map = Basemap(resolution=resolution,llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
      
      
      x = np.linspace(llcrnrlon,urcrnrlon,N_column,endpoint=True)
      x_value = x + (x[1]-x[2])/2.0
      x_value = x_value[:-1]

      y = np.linspace(llcrnrlat,urcrnrlat,N_row,endpoint=True)
      y_value = y + (y[1]-y[2])/2.0
      y_value = y_value[:-1]

      mask = np.ones((N_row-1,N_column-1),dtype=int)

      print("N_row = ",N_row)
      for k in range (len(x_value)):
          print("k = ",k)
          for i in range(len(y_value)):
              if map.is_land(x_value[k],y_value[i]):
                 mask[i,k] = 0

      #mask = mask[::-1,::-1]
      mask[0,:] = 0
      mask[-1,:] = 0
      mask[:,0] = 0
      mask[:,-1] = 0
      
      if plot_img:
         map.drawcoastlines()
         map.imshow(mask)
         map.drawcoastlines()
         plt.show() 

      return mask


  def findEdges(self,mask,plot_img=False):
      mask_rolled = np.roll(mask,1,axis=1)
      edges_column = np.absolute(mask - mask_rolled)
      mask_rolled = np.roll(mask,1,axis=0)
      edges_row = np.absolute(mask - mask_rolled)
      edges = edges_row + edges_column

      edges[edges>1] = 1
      #TODO: Take into account -1 offset, land water difference
      #xy = np.asarray(np.where(edges_column == 1)).T
      #edges_column[edges_column==1] = 0
      #xy[:,0] = xy[:,0]+1  

      edges[0,:] = 0
      edges[:,0] = 0
      edges[1,:] = 0
      edges[:,1] = 0
      
      edges[edges.shape[0]-1,:] = 0
      edges[:,edges.shape[1]-1] = 0
      edges[edges.shape[0]-2,:] = 0
      edges[:,edges.shape[1]-2] = 0

      if plot_img:
         plt.imshow(edges[::-1,:])
         plt.show()
      return edges

  def expandMask(self,edges,window,plot_img=False):
      xy = np.asarray(np.where(edges == 1)).T
      mask = np.copy(edges)
      for k in range(xy.shape[0]):
          row = xy[k,0]
          column = xy[k,1]

          l_r = row - window
          if l_r < 0:
             l_r = 0
          u_r = row + window
          if u_r > edges.shape[0]-1:
             u_r = edges.shape[0]-1

          l_c = column - window
          if l_c < 0:
             l_c = 0
          u_c = column + window
          if u_c > edges.shape[1]-1:
             u_c = edges.shape[1]-1
          mask[l_r:u_r,l_c:u_c] = 1

      if plot_img:
         plt.imshow(mask[::-1,:])
         plt.show()
      return mask


      
  def shapely_test(self):
      from shapely.ops import polygonize
      import pprint
      #from descartes.patch import PolygonPatch 
      lines = [((0, 0), (1, 1)),
      ((0, 0), (0, 1)),
      ((0, 1), (1, 1)),
      ((1, 1), (1, 0)),
      ((1, 0), (0, 0))]

      p = list(polygonize(lines)) 
      pprint.pprint(list(polygonize(lines)))
      #fig = plt.figure(1, dpi=90)
      #ax = fig.add_subplot(122)
 
      

      for polygon in p:
          c = np.array(polygon.exterior.coords)
          print(c.shape)
          print(c)
          plt.plot(c[:,0],c[:,1])
          #plot_coords(ax, polygon.exterior)
          #patch = PolygonPatch(polygon, alpha=0.5, zorder=2)
          #ax.add_patch(patch)

      plt.show()

  def test_poly_mask(self):
      import pylab as plt
      import numpy as np
      from matplotlib.path import Path

      width, height=2000, 2000

      polygon=[(0.1*width, 0.1*height), (0.15*width, 0.7*height), (0.8*width, 0.75*height), (0.72*width, 0.15*height)]
      poly_path=Path(polygon)

      x, y = np.mgrid[:height, :width]
      coors=np.hstack((x.reshape(-1, 1), y.reshape(-1,1))) # coors.shape is (4000000,2)

      mask = poly_path.contains_points(coors)
      plt.imshow(mask.reshape(height, width))
      plt.show()

  def plotEU(self,file_save='TwoDGridNARI.sav',N=10001,llcrnrlon=0,llcrnrlat=35,urcrnrlon=15,urcrnrlat=45,plot_img=True):
      if plot_img:
         map = Basemap(resolution='h',llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat)
         map.drawcoastlines()

      x = np.linspace(-180,180,N,endpoint=True)
      x_value = x + (x[1]-x[2])/2.0
      x_value = x_value[:-1]
      
      y = np.linspace(-90,90,N,endpoint=True)
      y_value = y + (y[1]-y[2])/2.0
      y_value = y_value[:-1]

      index_x_1 = np.searchsorted(x,llcrnrlon)-1
      index_x_2 = np.searchsorted(x,urcrnrlon)-1

      index_y_1 = np.searchsorted(y,llcrnrlat)-1
      index_y_2 = np.searchsorted(y,urcrnrlat)-1

      matriks = joblib.load(file_save)
      sub_m = matriks[index_y_1:index_y_2,index_x_1:index_x_2]
      matriks=''
      
      sub_m = np.log(sub_m+1)
      if plot_img:
         map.imshow(sub_m)
         plt.show()

      return sub_m.shape

  def load_config_file(self,file_name="AGEAN.ini"):
      config = configparser.ConfigParser()
      print(file_name)
      config.read(file_name)
      sections = config.sections() 
      print(sections)
      parameter_dictionary = {}
      '''      
      llcrnrlon = 22
      llcrnrlat = 30
      urcrnrlon = 30 
      urcrnrlat = 42
      resolution = h

      [FILENAMES]
      global_heatmap = TwoDGrid.sav
      N = 10001
      coast_line_window = 3
      large_coast_line_mask= mask.sav
      water_mask = water_mask.sav

      [GENERAL PARAMETERS]
      m_size = 5
      threshold=150
      o_size = 3

      [HOUGH TRANSFORM PARAMETERS]
      min_distance=18
      min_angle=20 
      h_threshold=0.6
      num_peaks=10
      '''
      int_list = ["N","m_size","threshold","o_size","num_peaks","coast_line_window","min_distance","min_angle"]
      float_list = ["h_threshold","num_peaks","llcrnrlon","llcrnrlat","urcrnrlon","urcrnrlat","urcrnrlat"]
     
      for s in sections:
          for p in config[s]:
              if p in int_list:   
                 parameter_dictionary[str(p)] = int(config[s][p])
              elif p in float_list:
                 parameter_dictionary[str(p)] = float(config[s][p])
              else:
                 parameter_dictionary[str(p)] = str(config[s][p]) 

      return parameter_dictionary
        
if __name__ == "__main__":
   s = HmapSeg()
   s.plot_play()
   #s.create_Dictionary_NARI()
   #s.gridData_to_2DMap_NARI(file_name="nari_dynamic.csv",N=10001,v=1000000,file_save='TwoDGridNARI.sav')
   #s.plotEU(file_save='TwoDGrid.sav',N=10001,llcrnrlon=-10,llcrnrlat=45,urcrnrlon=0,urcrnrlat=51,plot_img=True)
   '''
   config_file = sys.argv[1]
   s = HmapSeg()
   #s.drawWorldMap()
   d_parm = s.load_config_file(file_name=config_file)
   print(d_parm)
   #LOADING PARAMETERS FROM CONFIGURATION FILE
   print("LOADING PARAMETERS FROM FILE: "+config_file)
   
   #GENERATE MASKS IF NEEDED
   if not os.path.isfile("./"+d_parm["large_coast_line_mask"]):

      dim = s.plotEU(file_save=d_parm["global_heatmap"],N=d_parm["n"],llcrnrlon=d_parm["llcrnrlon"],llcrnrlat=d_parm["llcrnrlat"],urcrnrlon=d_parm["urcrnrlon"],urcrnrlat=d_parm["urcrnrlat"],plot_img=False)
      print("CREATING LAND WATER MASK: "+d_parm["large_coast_line_mask"])
      mask = s.createMask(llcrnrlon=d_parm["llcrnrlon"],llcrnrlat=d_parm["llcrnrlat"],urcrnrlon=d_parm["urcrnrlon"],urcrnrlat=d_parm["urcrnrlat"],N_row=dim[0]+1,N_column=dim[1]+1,resolution=d_parm['resolution']) #CREATE WATER LAND MASK
      joblib.dump(mask, d_parm["water_mask"]) 
      print("CREATING COASTLINE MASK: "+d_parm["large_coast_line_mask"]) 
      edges = s.findEdges(mask) #FIND COASTLINE
      e_mask = s.expandMask(edges,d_parm["coast_line_window"]) #GENERATE COASTLINE MASK
      joblib.dump(e_mask, d_parm["large_coast_line_mask"]) 

   #RUN SEGMENTATION ALGORITHM
   print("RUN SEGMENTATION ALGORITHM")  
   s.polygonSegmentation(file_save=d_parm["global_heatmap"],N=d_parm["n"], mask_file=d_parm["large_coast_line_mask"], water_mask = d_parm["water_mask"], resolution=d_parm["resolution"],llcrnrlon=d_parm["llcrnrlon"],llcrnrlat=d_parm["llcrnrlat"],urcrnrlon=d_parm["urcrnrlon"],urcrnrlat=d_parm["urcrnrlat"], m_size = d_parm["m_size"], threshold=d_parm["threshold"], o_size = d_parm["o_size"], min_distance=d_parm["min_distance"], min_angle=d_parm["min_angle"], h_threshold=d_parm["h_threshold"],num_peaks=d_parm["num_peaks"],config_file=config_file)  
   '''
   #print(dim)
   #print(d_parm)
   #s.polygonSegmentation()
   #s.test_poly_mask()
   #s.testMedian()
   #mask = s.createMask()
   #joblib.dump(mask, "water_mask.sav") 
   #edges = s.findEdges(mask)
   #e_mask = s.expandMask(edges,3)
   #joblib.dump(e_mask, "mask.sav") 
   #x = np.diag(np.ones((8,),dtype=int)) 
   #m,p = s.followLine(3,5,x+1)
   #print(p)
   #plt.imshow(m)
   #plt.show()
   
   #s.plotGriddedData()
       

'''
#OLD CODE
filename = 'finalized_model.sav'
loaded_model = joblib.load(filename)
plt.imshow(loaded_model[::-1,:],vmax=1000,cmap=plt.cm.get_cmap('Reds'))

'''
'''
map = Basemap()

map.drawcoastlines()

N = 10001
x = np.linspace(-180,180,N,endpoint=True)
x_value = x + (x[1]-x[2])/2.0
x_value = x_value[:-1]

y = np.linspace(-90,90,N,endpoint=True)
y_value = y + (y[1]-y[2])/2.0
y_value = y_value[:-1]

for k in range(N-1):
  print("k = "+str(k))
  for l in range(N-1):
    if loaded_model[k,l]>100:
      map.scatter(x_value[k],y_value[l],s=0.01)
      
plt.show()      
'''
'''

#loaded_model = loaded_model.astype(float)
#print(loaded_model.shape)

#plt.imshow(loaded_model[:4000,:4000],vmax=10);


#x1 = np.linspace(0, 10, 8, endpoint=True)
#print(x1)

#index = np.searchsorted(x1,1)
#print(index)
'''
'''
N = 10001
x = np.linspace(-180,180,N,endpoint=True)
x_value = x + (x[1]-x[2])/2.0
x_value = x_value[:-1]

y = np.linspace(-90,90,N,endpoint=True)
y_value = y + (y[1]-y[2])/2.0
y_value = y_value[:-1]

matriks = np.zeros((N-1,N-1),dtype=int)

counter = 0

with open("8monthsAIS.txt") as infile:
    for line in infile:
        line_split = line.split(" ")

        lat = float(line_split[2]) 
        lon = float(line_split[3])

        index_x = np.searchsorted(x,lon)
        index_y = np.searchsorted(y,lat)

        if (index_x == N):
           index_x = index_x - 2
        elif (index_x > 0):
           index_x = index_x - 1
           
        if (index_y == N):
           index_y = index_y - 2
        elif (index_y > 0):
           index_y = index_y - 1
 
        matriks[index_y,index_x]+=1

        counter = counter + 1
        if (counter % 1000000 == 0):
           #break
           print("counter = ",counter) 
#import dill as pickle         

# Pickle the 'data' dictionary using the highest protocol available.
#pickle.dump(matriks, open('data.pickle', 'wb'))

from sklearn.externals import joblib
filename = 'finalized_model.sav'
joblib.dump(matriks, filename)  

#with open('data.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
#    data = pickle.load(f)

plt.imshow(matriks[::-1,:],extent=[-180,180,-90,90],vmax=30)
plt.grid('on')
plt.show()
#plt.get_cmap('binary')

map = Basemap()

map.drawcoastlines()

#plt.show()
'''

'''
fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.8])
# setup mercator map projection.
m = Basemap(llcrnrlon=-100.,llcrnrlat=20.,urcrnrlon=20.,urcrnrlat=60.,\
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',projection='merc',\def plotEU(self,file_save='TwoDGrid2.sav',vmax=5000,cmv='hot',image_name='test5.pdf',N=10001):
            lat_0=40.,lon_0=-20.,lat_ts=20.)
# nylat, nylon are lat/lon of New York
nylat = 40.78; nylon = -73.98
# lonlat, lonlon are lat/lon of London.
lonlat = 51.53; lonlon = 0.08
# draw great circle route between NY and London
m.drawgreatcircle(nylon,nylat,lonlon,lonlat,linewidth=2,color='b')
m.drawcoastlines()
m.fillcontinents()
# draw parallels
m.drawparallels(np.arange(10,90,20),labels=[1,1,0,1])
# draw meridians
m.drawmeridians(np.arange(-180,180,30),labels=[1,1,0,1])
ax.set_title('Great Circle from New York to London')
'''
'''
counter = 0
mem_counter = 0

lons = np.zeros((10,),dtype=float)
lats = np.zeros((10,),dtype=float)

#lons_t = [-180, 180, -20.47, -20]
#lats_t = [90, -90, 40, -20]

#x, y = map(lons_t, lats_t)

#map.scatter(x, y, marker='o',color='r',s=10)
#map.scatter(lons_t, lats_t, marker='o',color='r',s=10)

#print("x = "+str(x))
#print("y = "+str(y))

start = timeit.timeit()




with open("8monthsAIS.txt") as infile:
    for line in infile:
        line_split = line.split(" ")

        #lats[mem_counter] = float(line_split[2]) 
        #lons[mem_counter] = float(line_split[3])

        #mem_counter = mem_counter+1

        #if mem_counter == 10:
           #lons = [0, 10, -20, -20]
           #lats = [0, -10, 40, -20]

           #x, y = map(lons, lats)

           #map.scatter(x, y, marker='o',color='g',s=10)
           #mem_counter = 0

        print(line_split)
        if (counter%1000000) == 0:
           print("counter = "+str(counter))
        counter = counter + 1
        if counter == 100:
           break

infile.close()
end = timeit.timeit()
print(end - start)

plt.show()
plt.savefig('test.png')

      
      sub_m_copy[sub_m<2] = 0
      
      X = np.reshape(sub_m_copy,(sub_m_copy.shape[0]*sub_m_copy.shape[1],1))
      
      from sklearn.cluster import KMeans
      
      kmeans = KMeans(n_clusters=7).fit(X)
      
      #KMeans(n_clusters=2, random_state=0).fit(X)
      
      
      sub_m_copy = np.reshape(kmeans.labels_,(sub_m_copy.shape[0],sub_m_copy.shape[1]))
      
      sub_m = np.copy(sub_m_copy)
      
      sub_m_copy[sub_m==6] = 1
      sub_m_copy[sub_m<>6] = 0
            
      #from skimage import filters
      #edges = filters.sobel(sub_m)
      
      #hist = np.histogram(sub_m.flatten(), bins='auto')

      im = map.imshow(sub_m_copy,cmap=plt.cm.get_cmap(cmv))
      cb = map.colorbar(im, location='bottom', label="contour lines")
      plt.show()
      plt.close()
      
      from matplotlib import cm
      from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
      sub_m_copy = sub_m_copy[::-1,:]
      # Classic straight-line Hough transform
      h, theta, d = hough_line(sub_m_copy)
      
      plt.imshow(np.log(1 + h),extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],cmap=cm.gray)#aspect=1/1.5
      plt.axes().set_aspect('auto', adjustable='box')
      plt.show()
      plt.close()
      
     
      for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
          y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
          y1 = (dist - sub_m_copy.shape[1] * np.cos(angle)) / np.sin(angle)
          plt.plot((0, sub_m_copy.shape[1]), (y0, y1), '-r')
      plt.imshow(sub_m_copy,cmap=plt.cm.get_cmap(cmv))
      plt.axes().set_aspect('auto', adjustable='box')
      plt.show()
      plt.close()
      
      lines = probabilistic_hough_line(sub_m_copy, threshold=10, line_length=50,
                                 line_gap=10)
      
      for line in lines:
          p0, p1 = line
          plt.plot((p0[0], p1[0]), (p0[1], p1[1]))

      plt.imshow(sub_m_copy,cmap=plt.cm.get_cmap(cmv))
      plt.axes().set_aspect('auto', adjustable='box')
      plt.show()
      
      #map.colorbar(im)
      
      plt.savefig(image_name)
      
      joblib.dump(sub_m, "EU2.sav")   
'''
'''
      #for polygon in p:
      #    c = np.array(polygon.exterior.coords)
      #    print(c.shape)
      #    print(c)
      #    plt.plot(c[:,0],c[:,1])
          #plot_coords(ax, polygon.exterior)
          #patch = PolygonPatch(polygon, alpha=0.5, zorder=2)
          #ax.add_patch(patch)

      #plt.show()

      

      #eroded = erosion(med,selem)
      #selem = square(2)
      #eroded = erosion(med,selem)
      #map.imshow(eroded)
      #plt.show()
      
      #med = self.convertMatToGray(med)
      #map.imshow(med)
      #plt.show()

      #from skimage.morphology import extrema
      #maxima = extrema.h_maxima(med,50)
      #map.drawcoastlines()
      #map.imshow(maxima,cmap=cm.jet)
      #plt.show()

      #fig, ax = plt.subplots() 
      #blobs_doh = blob_doh(med, max_sigma=30, threshold=.01)
      #ax.imshow(med) 
      #for blob in blobs_doh:
      #    print(blob)
      #    y, x, r = blob
      #    c = plt.Circle((x, y), r, color="r", linewidth=5, fill=False)
      #    ax.add_patch(c)
      #plt.show()
      #med = self.convertMatToGray(med)
      #labels = segmentation.slic(med, compactness=0.01, n_segments=3,enforce_connectivity=False)
      #labels = labels + 1
      #map.imshow(labels)
      #out1 = color.label2rgb(labels, med, kind='avg') 
      #map.imshow(out1)
      #plt.show()
      #map.drawcoastlines()
      #rag = graph.rag_mean_color(med, labels, mode="similarity")

      #labels2 = graph.cut_normalized(labels, rag)
      #out2 = color.label2rgb(labels2, med, kind='avg')




      #print("hallo="+str(np.amin(med)))
      #print("hallo="+str(np.amax(med)))  
      
      #label_rgb = color.label2rgb(labels, med, kind='avg')
      #map.imshow(med)
      
      #med[sub_m_old==0] = 0
      #med = med>50
      #med[sub_m_old==0]=0
      #med = med*m 

      #from matplotlib import cm
      #map.imshow(med,cmap=cm.hot)
      #plt.show()
'''
'''
      #histo = plt.hist(np.absolute(sub_m.ravel()-med_old.ravel())*m.ravel(), bins=np.arange(0, 256))
      #plt.show() 

      
      #med = filters.gaussian(med, sigma=0.5)
      #med = med<0.5
       
      from skimage.morphology import erosion, dilation, opening, closing, white_tophat,binary_opening,thin
      from skimage.morphology import black_tophat, skeletonize, convex_hull_image
      from skimage.morphology import disk,square
      selem = disk(1)
      eroded = erosion(med,selem)

      #med = filters.sobel(med)
      #med = med*m
      
      #med = (med-np.amax(med))*(-1)
      #med[sub_m_old == 0] = 0
      #med = med*m
      #from matplotlib import cm
      #med = self.convertMatToGray(med) 
      #from skimage import feature
      #edges1 = feature.canny(med,sigma=2) 
      #med_t = med>50
      #eroded = eroded<0.5
      #eroded = filters.gaussian(eroded, sigma=1.0)
      map.imshow(eroded,cmap=cm.gray)
      print(np.amin(eroded))
      plt.show()
      histo = plt.hist(eroded, bins=np.arange(0, 256))
      plt.show() 

      from skimage.filters import try_all_threshold

      #eroded = self.convertMatToGray(eroded)

      

      fig, ax = try_all_threshold(eroded.ravel(), figsize=(10, 8), verbose=False)
      plt.show()

      #med[m_old==1] = 0
      #map.imshow(np.absolute(med))
      
      xx,yy = np.meshgrid(x_value[index_x_1:index_x_2],y_value[index_y_1:index_y_2])
      #from scipy import interpolate  
      #f = interpolate.interp2d(x_value[index_x_1:index_x_2], y_value[index_y_1:index_y_2], med, kind='cubic')
      #x_new = np.linspace(x_value[index_x_1],x_value[index_x_2],2000)
      #y_new = np.linspace(y_value[index_y_1],y_value[index_y_2],2000) 
      #xx_new,yy_new = np.meshgrid(x_new,y_new)
      #m_new = f(x_new,y_new)
      

      #cs = map.contourf(xx,yy,med,3)
      #map.imshow(med,interpolation="nearest")
      #plt.show()
      #plt.show() 
      #map.drawcoastlines()  
      #med[med==0] = 1

      X = np.reshape(med,(med.shape[0]*med.shape[1],1))

      X2 = X[X<>0]

      X2 = np.reshape(X2,(len(X2),1))

      
      from sklearn.cluster import KMeans
      
      kmeans = KMeans(n_clusters=3, random_state=0).fit(X2)
      
      #KMeans(n_clusters=2, random_state=0).fit(X)
      
      
      X[X<>0] = kmeans.labels_+1   
  
      #X = X==2

      med = np.reshape(X,(med.shape[0],med.shape[1])) 

      #med = med < 0.4
      #plt.show()
      #v = np.absolute(med.ravel())
      #v = v[v<>0]
      #histo = plt.hist(v)
      #plt.show()
      #sub_m = sub_m*m
      #from skimage import filters
      #med = filters.sobel(med)
      
      #med = median(med, disk(1))
      #med[med==0] = 255
      #med = med<150
      
      #map.imshow(np.absolute(med))
      #plt.show() 
      map.drawcoastlines()
      cs = map.contour(xx,yy,med,3)
      plt.show()
      
      map.drawcoastlines()

      c_v = ["b","r","m","c","g","k","b","r"]

      for i in range(1,2):

          paths = cs.collections[i].get_paths()
          k = 0      

          for k in range(len(paths)):
              v = paths[k].vertices
              if len(v[:,0]) > 50:
           
                 x = v[:,0]
                 y = v[:,1]

                 #hull = ConvexHull(v)
                 #for simplex in hull.simplices:
                 #plt.plot(v[simplex, 0], v[simplex, 1], c=c_v[4])

                 map.plot(x,y,c=c_v[i])
                 #plt.Polygon(segments[0], fill=False, color='w')
              k = k+1
              if k%100 == 0:
                print(str(k))
                print(str(len(paths)))
      plt.show()
      

      from matplotlib import cm
      from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
      sub_m_copy = med[::-1,:]
      # Classic straight-line Hough transform
      h, theta, d = hough_line(sub_m_copy)
      
      plt.imshow(np.log(1 + h),extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],cmap=cm.gray)#aspect=1/1.5
      plt.axes().set_aspect('auto', adjustable='box')
      plt.show()
      plt.close()
      
     
      #map.drawcoastlines()
      S_land = []
      for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=10, min_angle=20, threshold=0.6*np.max(h),num_peaks=10)):#numpeaks
          y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
          print("y0 = "+str(y0))

          y1 = (dist - sub_m_copy.shape[1] * np.cos(angle)) / np.sin(angle)
          print("y1 = "+str(y1))
          S_land.append(((0,sub_m_copy.shape[1]), (y0, y1)))
          plt.plot((0, sub_m_copy.shape[1]), (y0, y1), '-r')
          #break
      plt.imshow(sub_m_copy,cmap=plt.cm.get_cmap(cmv))
      #plt.axes().set_aspect('auto', adjustable='box')
      plt.show()
      plt.close() 
      
      
      #histo = plt.hist(np.absolute(med.ravel()), bins=np.arange(0, 256))
      #plt.show()
      #sub_m = sub_m*m
'''

