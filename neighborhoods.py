from factual import Factual
from prettytable import PrettyTable
import googlemaps
from factual.utils import circle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


def GetLatLng( address ):
	gmaps = googlemaps.Client(key="AIzaSyCkB6R2M0-qEXXtTMOjqxDfsmPkb-v0oB0")
	# Geocoding an address
	result = gmaps.geocode( address )[0]["geometry"]["location"]
	return result["lat"], result["lng"]

def GetBizNearAddress( category_id, address, radius_meters, offset, n ):
	factual = Factual('YK4fO9P1qZFZy4Cu7HDNkeZd9YEN1ut4tyO426OR', '9G1L9zUop7kwMmNkM7eIYsFNbvhINDl8hyxyZnKv')
	places = factual.table('places')
	# All with category_id
	rest = places.filters({"category_ids":{"$includes": category_id}})
	# Latitutde and Longitude of address
	lat, lng = GetLatLng( address )
	near = rest.geo(circle(lat, lng, radius_meters)).offset(offset).limit(n).data()
	return near

##-------##-------##-------##-------##-------##-------##-------##-------##

address = "1842 Purdue Ave, 90025"
#address = '217 Commonwealth Avenue, 02116'
#address = '4555 Dean Martin Drive, 89103'
#address = '159 West 48th Street, 10036'
category_id = 334 #bars=312
offset = 0
near = []
n = 50
random_state = 170 #random state for k means alg

for x in xrange(1,10):
	near += GetBizNearAddress( category_id, address, 20000, offset, n )
	offset += n

X = np.empty([1,2])

for r in near:

	datapoint = np.array([[r['longitude'],r['latitude']]])
	X = np.append(X, datapoint, axis=0)

X = X[1:] #remo

maxClusters = 10
errors = []

for nCl in xrange(2,maxClusters+1):
	clustering = KMeans(n_clusters=nCl, random_state=random_state).fit(X)
	centers = clustering.cluster_centers_
	clusters = clustering.predict(X)

	mse = 0
	for point, cluster in zip(X,clusters):
		mse += sum(pow(point - centers[cluster],2))

	errors.append(mse)


plt.plot(xrange(2,maxClusters+1),errors,'ro')
plt.show()

bestClusterNumber = 8

y_pred = KMeans(n_clusters=bestClusterNumber,random_state=random_state).fit_predict(X)
plt.scatter(X[:,0],X[:,1], c=y_pred)
plt.show()



