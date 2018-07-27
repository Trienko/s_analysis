from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import timeit
from sklearn.externals import joblib
import matplotlib.pyplot as plt

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
   s.plotEU()
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
