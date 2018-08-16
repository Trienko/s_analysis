from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import timeit
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

class SClass():

  def __init__(self):
      pass

  def drawWorldMap(self):
      map = Basemap()
      map.drawcoastlines()
      map.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0])
      map.drawmeridians(np.arange(-180,181,30), labels=[0,0,0,1])
      plt.show()

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

  def convertMatToGray(self,matA):
      max_v = np.max(matA)
      fac = 255.0/max_v
      matA = np.round(matA*fac).astype(np.uint8)
      return matA

  def applyKMeans(self,file_save='TwoDGrid.sav',cmv='hot',N=10001):
      map = Basemap(resolution='h',llcrnrlon=22, llcrnrlat=30,urcrnrlon=30, urcrnrlat=42)
      map.drawcoastlines()

      x = np.linspace(-180,180,N,endpoint=True)
      x_value = x + (x[1]-x[2])/2.0
      x_value = x_value[:-1]

      y = np.linspace(-90,90,N,endpoint=True)
      y_value = y + (y[1]-y[2])/2.0
      y_value = y_value[:-1]

      index_x_1 = np.searchsorted(x,22)-1
      index_x_2 = np.searchsorted(x,30)-1

      index_y_1 = np.searchsorted(y,30)-1
      index_y_2 = np.searchsorted(y,42)-1

      matriks = joblib.load(file_save)
      sub_m = matriks[index_y_1:index_y_2,index_x_1:index_x_2]
      matriks=''
      
      sub_m = np.log(sub_m)
      sub_m = self.convertMatToGray(sub_m)

      im = map.imshow(sub_m,cmap=plt.cm.get_cmap('gray'))
      plt.show()

      X = np.reshape(sub_m,(sub_m.shape[0]*sub_m.shape[1],1))


      X2 = X[X<>0]

      X2 = np.reshape(X2,(len(X2),1))

      
      from sklearn.cluster import KMeans
      
      kmeans = KMeans(n_clusters=5, random_state=0).fit(X2)
      
      #KMeans(n_clusters=2, random_state=0).fit(X)
      
      
      X[X<>0] = kmeans.labels_+1   

      sub_m_copy = np.reshape(X,(sub_m.shape[0],sub_m.shape[1]))
      map = Basemap(resolution='h',llcrnrlon=22, llcrnrlat=30,urcrnrlon=30, urcrnrlat=42)
      map.drawcoastlines()
      #threshold 
      sub_copy
      map.imshow(sub_m_copy)
      plt.show() 

      

  def followLine(self, p, q, two_dim_array, max_points = 100):
      processed_points = np.zeros((1,2),dtype = int)
      processed_points[0,0] = p
      processed_points[0,1] = q

      mask = np.zeros(two_dim_array.shape,dtype=int)
      mask[p,q] = 1
      
      r = p
      c = q

      x_max = two_dim_array.shape[0]
      y_max = two_dim_array.shape[1]

      points = np.array([])
      
      if two_dim_array[p,q] == 0:
         return mask,processed_points

      if (p == 0) or (q == 0):
         return mask,processed_points

      if (p == x_max - 1) or (q == y_max - 1):
         return mask,processed_points

      continue_var = True
      points_counter = 0

      while (continue_var) and (points_counter <= max_points):
           
            one = np.absolute(two_dim_array[r,c] - two_dim_array[r-1,c-1])
            two = np.absolute(two_dim_array[r,c] - two_dim_array[r-1,c])
            three = np.absolute(two_dim_array[r,c] - two_dim_array[r-1,c+1])
            four = np.absolute(two_dim_array[r,c] - two_dim_array[r,c-1])
            five = np.absolute(two_dim_array[r,c] - two_dim_array[r,c+1])
            six = np.absolute(two_dim_array[r,c] - two_dim_array[r+1,c-1])
            seven = np.absolute(two_dim_array[r,c] - two_dim_array[r+1,c])
            eight = np.absolute(two_dim_array[r,c] - two_dim_array[r+1,c+1])

            idx_array = np.array([[r-1,c-1],[r-1,c],[r-1,c+1],[r,c-1],[r,c+1],[r+1,c-1],[r+1,c],[r+1,c+1]])

            temp_values = np.array([one,two,three,four,five,six,seven,eight])

            idx_sort = np.argsort(temp_values) 
            idx_array = idx_array[idx_sort,:]

            temp_points = np.zeros((2,2),dtype = int)

            temp_points[0,:] = idx_array[0,:]
            temp_points[1,:] = idx_array[1,:]
 
            for k in range(2):
                temp_points[k,:] = idx_array[k,:]
                if ((temp_points[k,0] <> 0) and (temp_points[k,1] <> 0) and (temp_points[k,0] <> x_max-1) and (temp_points[k,0] <> y_max-1)):
                   if (mask[temp_points[k,0],temp_points[k,1]] <> 1):
                      if points.size == 0:
                         points = np.zeros((1,2),dtype = int)
                         points[0,0] = temp_points[k,0]
                         points[0,1] = temp_points[k,1]
                      else:
                         points = np.vstack((temp_points[k,:],points))  
                      mask[temp_points[k,0],temp_points[k,1]] = 1    
                else:
                    processed_points = np.vstack((processed_points,temp_points[k,:]))
                    mask[temp_points[k,0],temp_points[k,1]] = 1 

                
            
            temp_points = np.zeros((1,2),dtype = int)
            
            temp_points[0,0] = r
            temp_points[0,1] = c
            
            if points_counter <> 0:
               processed_points = np.vstack((processed_points,temp_points))
               mask[temp_points[0,0],temp_points[0,1]] = 1
            
            if points.size == 0:
               continue_var = False
            else:
               r = points[0,0]
               c = points[0,1]
               if points.size > 1: 
                  points = points[1:,:]
               else:
                  points = np.array([])

            points_counter = points_counter + 1
            print(mask)
            print(processed_points)
            print(points)

     
      return mask, processed_points      
  
  def testSmallerImage(self,file_save='TwoDGrid.sav',cmv='hot',N=10001):
      map = Basemap(resolution='h',llcrnrlon=22, llcrnrlat=30,urcrnrlon=30, urcrnrlat=42)
      map.drawcoastlines()

      x = np.linspace(-180,180,N,endpoint=True)
      x_value = x + (x[1]-x[2])/2.0
      x_value = x_value[:-1]

      y = np.linspace(-90,90,N,endpoint=True)
      y_value = y + (y[1]-y[2])/2.0
      y_value = y_value[:-1]

      index_x_1 = np.searchsorted(x,22)-1
      index_x_2 = np.searchsorted(x,30)-1

      index_y_1 = np.searchsorted(y,30)-1
      index_y_2 = np.searchsorted(y,42)-1

      matriks = joblib.load(file_save)
      sub_m = matriks[index_y_1:index_y_2,index_x_1:index_x_2]
      matriks=''
      
      sub_m = np.log(sub_m)
      sub_m = self.convertMatToGray(sub_m)

      m,p = self.followLine(469,81,sub_m.astype(int))
      

      im = map.imshow(sub_m,cmap=plt.cm.get_cmap('gray'))
      plt.show()

      plt.imshow(sub_m)
      plt.show()

      plt.imshow(m)
      plt.show()
      
      histo = plt.hist(sub_m.ravel(), bins=np.arange(0, 256))
      plt.show() 

      print(sub_m.shape)

      plt.plot(sub_m[300,:])
      plt.plot(sub_m[301,:])
      plt.plot(sub_m[302,:])
      plt.show()


      from skimage.filters import threshold_otsu

      thresh = threshold_otsu(sub_m)
      binary = sub_m > 140

      plt.imshow(binary[:,::-1],aspect='auto',cmap=plt.cm.get_cmap('gray'))
      plt.show()

      
      '''
      #new_m = np.zeros(sub_m.shape,dtype=int)
      #new_m[sub_m>75] = 1
      #from skimage.feature import canny
      #from skimage.morphology import watershed, disk
      #from skimage import data
      #from skimage.filters import rank
      #from skimage.util import img_as_ubyte
      
      #denoised = rank.median(sub_m, disk(2)) 
      
 

      #print(im.shape)
      
      #from skimage import measure

      #contours = measure.find_contours(sub_m/255.0, 0.5)
      #for n, contour in enumerate(contours):
      #    map.plot(contour[:, 1], contour[:, 0], linewidth=3, c="b")
      #plt.show()

      edges = canny(denoised/255.,sigma=1.0)
      plt.imshow(edges[::-1,:],cmap=plt.cm.get_cmap('gray'),aspect='auto')
      #print(edges)
      plt.show()

      #from scipy import ndimage as ndi
      #fill_coins = ndi.binary_fill_holes(edges)
      #plt.imshow(fill_coins[::-1,:],cmap=plt.cm.get_cmap('gray'),aspect='auto')
      #print(edges)
      #plt.show()


      histo = plt.hist(sub_m.ravel(), bins=np.arange(0, 256))
      plt.show()

      #from skimage.segmentation import slic

      #segments = slic(sub_m/255., n_segments=1000, compactness=10)
      #plt.imshow(segments,cmap=plt.cm.get_cmap('grey'),aspect='auto')
      #plt.show()
      '''

      '''
      from skimage.filters import threshold_otsu, threshold_local

      image = np.copy(sub_m)

      global_thresh = threshold_otsu(image)
      binary_global = image > global_thresh

      block_size = 75
      local_thresh = threshold_local(image, block_size, offset=10)
      binary_local = image > local_thresh

      plt.imshow(binary_global[::-1,:],cmap=plt.cm.get_cmap('gray'),aspect='auto')
      plt.show()
      plt.imshow(binary_local[::-1,:],cmap=plt.cm.get_cmap('gray'),aspect='auto')
      plt.show() 

      from skimage.morphology import disk
      from skimage.filters import threshold_otsu, rank
      from skimage.util import img_as_ubyte
      
      img = img_as_ubyte(image)

      radius = 20
      selem = disk(radius)

      local_otsu = rank.otsu(img, selem)

      plt.imshow(img[::-1,:]>local_otsu[::-1,:],cmap=plt.cm.get_cmap('gray'),aspect='auto')
      plt.show()  

      from skimage.filters import unsharp_mask
      result_1 = unsharp_mask(image, radius=1, amount=1)
      plt.imshow(result_1[::-1,:]>local_otsu[::-1,:],cmap=plt.cm.get_cmap('gray'),aspect='auto')
      plt.show()  
      '''


  def plotEUGray(self,file_save='TwoDGrid.sav',cmv='hot',N=10001):
      map = Basemap(resolution='h',llcrnrlon=-15, llcrnrlat=30,urcrnrlon=30, urcrnrlat=60)
      map.drawcoastlines()

      x = np.linspace(-180,180,N,endpoint=True)
      x_value = x + (x[1]-x[2])/2.0
      x_value = x_value[:-1]

      y = np.linspace(-90,90,N,endpoint=True)
      y_value = y + (y[1]-y[2])/2.0
      y_value = y_value[:-1]

      index_x_1 = np.searchsorted(x,-15)-1
      index_x_2 = np.searchsorted(x,30)-1

      index_y_1 = np.searchsorted(y,30)-1
      index_y_2 = np.searchsorted(y,60)-1

      matriks = joblib.load(file_save)
      sub_m = matriks[index_y_1:index_y_2,index_x_1:index_x_2]
      matriks=''
      
      sub_m = np.log(sub_m)
      sub_m = self.convertMatToGray(sub_m)

      #new_m = np.zeros(sub_m.shape,dtype=int)
      #new_m[sub_m>75] = 1
      from skimage.feature import canny
      from skimage.morphology import watershed, disk
      from skimage import data
      from skimage.filters import rank
      from skimage.util import img_as_ubyte
      
      denoised = rank.median(sub_m, disk(2)) 
      
      im = map.imshow(denoised,cmap=plt.cm.get_cmap('gray'))
      plt.show()

      #print(im.shape)
      
      #from skimage import measure

      #contours = measure.find_contours(sub_m/255.0, 0.5)
      #for n, contour in enumerate(contours):
      #    map.plot(contour[:, 1], contour[:, 0], linewidth=3, c="b")
      #plt.show()

      edges = canny(denoised/255.,sigma=1.0)
      plt.imshow(edges[::-1,:],cmap=plt.cm.get_cmap('gray'),aspect='auto')
      #print(edges)
      plt.show()

      #from scipy import ndimage as ndi
      #fill_coins = ndi.binary_fill_holes(edges)
      #plt.imshow(fill_coins[::-1,:],cmap=plt.cm.get_cmap('gray'),aspect='auto')
      #print(edges)
      #plt.show()


      histo = plt.hist(sub_m.ravel(), bins=np.arange(0, 256))
      plt.show()

      #from skimage.segmentation import slic

      #segments = slic(sub_m/255., n_segments=1000, compactness=10)
      #plt.imshow(segments,cmap=plt.cm.get_cmap('grey'),aspect='auto')
      #plt.show()
      

      '''
      from skimage.filters import threshold_otsu, threshold_local

      image = np.copy(sub_m)

      global_thresh = threshold_otsu(image)
      binary_global = image > global_thresh

      block_size = 75
      local_thresh = threshold_local(image, block_size, offset=10)
      binary_local = image > local_thresh

      plt.imshow(binary_global[::-1,:],cmap=plt.cm.get_cmap('gray'),aspect='auto')
      plt.show()
      plt.imshow(binary_local[::-1,:],cmap=plt.cm.get_cmap('gray'),aspect='auto')
      plt.show() 

      from skimage.morphology import disk
      from skimage.filters import threshold_otsu, rank
      from skimage.util import img_as_ubyte
      
      img = img_as_ubyte(image)

      radius = 20
      selem = disk(radius)

      local_otsu = rank.otsu(img, selem)

      plt.imshow(img[::-1,:]>local_otsu[::-1,:],cmap=plt.cm.get_cmap('gray'),aspect='auto')
      plt.show()  

      from skimage.filters import unsharp_mask
      result_1 = unsharp_mask(image, radius=1, amount=1)
      plt.imshow(result_1[::-1,:]>local_otsu[::-1,:],cmap=plt.cm.get_cmap('gray'),aspect='auto')
      plt.show()  
      '''

  def createMask(self,llcrnrlon=-15, llcrnrlat=30,urcrnrlon=30, urcrnrlat=60,N=1001):
      map = Basemap(resolution='c',llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
      
      
      x = np.linspace(llcrnrlon,urcrnrlon,N,endpoint=True)
      x_value = x + (x[1]-x[2])/2.0
      x_value = x_value[:-1]

      y = np.linspace(llcrnrlat,urcrnrlat,N,endpoint=True)
      y_value = y + (y[1]-y[2])/2.0
      y_value = y_value[:-1]

      mask = np.ones((N-1,N-1),dtype=int)

      for k in range (len(x_value)):
          print("k = ",k)
          for i in range(len(y_value)):
              if map.is_land(x_value[k],y_value[i]):
                 mask[i,k] = 0

      #mask = mask[::-1,::-1]
      
      map.drawcoastlines()
      map.imshow(mask)
      map.drawcoastlines()
      plt.show() 

      return mask


  def findEdges(self,mask):
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

      plt.imshow(edges[::-1,:])
      plt.show()
      return edges

  def plotEUContour(self,file_save='TwoDGrid.sav',cmv='hot',N=10001):
      map = Basemap(resolution='h',llcrnrlon=-15, llcrnrlat=30,urcrnrlon=30, urcrnrlat=60)
      map.drawcoastlines()

      x = np.linspace(-180,180,N,endpoint=True)
      x_value = x + (x[1]-x[2])/2.0
      x_value = x_value[:-1]

      y = np.linspace(-90,90,N,endpoint=True)
      y_value = y + (y[1]-y[2])/2.0
      y_value = y_value[:-1]

      index_x_1 = np.searchsorted(x,-15)-1
      index_x_2 = np.searchsorted(x,30)-1

      index_y_1 = np.searchsorted(y,30)-1
      index_y_2 = np.searchsorted(y,60)-1

      matriks = joblib.load(file_save)
      sub_m = matriks[index_y_1:index_y_2,index_x_1:index_x_2]
      matriks=''
      
      sub_m = np.log(sub_m)
      im = map.imshow(sub_m,cmap=plt.cm.get_cmap(cmv))
      plt.show()

      xx,yy = np.meshgrid(x_value[index_x_1:index_x_2],y_value[index_y_1:index_y_2])

      cs = map.contourf(xx,yy,sub_m)
      #plt.show()
      
      map.drawcoastlines()

      c_v = ["b","r","m","c","g","k","b","r"]

      for i in range(3):

          paths = cs.collections[i].get_paths()
          k = 0      

          for k in range(len(paths)):
              v = paths[k].vertices
              if len(v[:,0]) > 500:
                 x = v[:,0]
                 y = v[:,1]

                 #hull = ConvexHull(v)
                 #for simplex in hull.simplices:
                 #    plt.plot(v[simplex, 0], v[simplex, 1], c=c_v[i])

                 map.plot(x,y,c=c_v[i])
                 #plt.Polygon(segments[0], fill=False, color='w')
              k = k+1
              if k%100 == 0:
                 print(str(k))
                 print(str(len(paths)))
      plt.show()
      
      #for k in range(1000):

      #    dat0= contours.allsegs[3][k]
      #    plt.plot(dat0[:,0],dat0[:,1],"r")

      #plt.show()

      


  def plotEU(self,file_save='TwoDGrid.sav',vmax=5000,cmv='hot',image_name='test5.pdf',N=10001):
      map = Basemap(resolution='h',llcrnrlon=-15, llcrnrlat=30,urcrnrlon=30, urcrnrlat=60)
      map.drawcoastlines()

      x = np.linspace(-180,180,N,endpoint=True)
      x_value = x + (x[1]-x[2])/2.0
      x_value = x_value[:-1]

      y = np.linspace(-90,90,N,endpoint=True)
      y_value = y + (y[1]-y[2])/2.0
      y_value = y_value[:-1]

      index_x_1 = np.searchsorted(x,-15)-1
      index_x_2 = np.searchsorted(x,30)-1

      index_y_1 = np.searchsorted(y,30)-1
      index_y_2 = np.searchsorted(y,60)-1

      matriks = joblib.load(file_save)
      sub_m = matriks[index_y_1:index_y_2,index_x_1:index_x_2]
      matriks=''
      
      sub_m = np.log(sub_m)
      
      sub_m_copy = np.copy(sub_m)
      
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
      
      joblib.dump(sub_m, "EU.sav")   
        
      

if __name__ == "__main__":
   s = SClass()
   #s.drawWorldMap()
   #s.plotEU()
   #s.plotEUGray()
   #s.applyKMeans()
   mask = s.createMask()
   s.findEdges(mask)
   #x = np.diag(np.ones((8,),dtype=int)) 
   #m,p = s.followLine(3,5,x+1)
   #print(p)
   #plt.imshow(m)
   #plt.show()
   
   #s.plotGriddedData()
   
      

'''
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
            resolution='l',projection='merc',\
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

'''
