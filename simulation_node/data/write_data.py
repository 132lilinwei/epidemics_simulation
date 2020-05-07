import random
import sys
import math

def write_graph(filepath, width, height, wgap, hgap):
	nnode = width * height
	edgeNum = 0
	hubList = generateHub(width, height, wgap, hgap)
	print("hubList num: ", len(hubList))
	computeEdgeNum = width * height * 4 - 2 * (width + height) + len(hubList) * (len(hubList) - 1)
	with open(filepath, 'w') as f:
		f.write(str(width) + " " + str(height) + " " + str(computeEdgeNum) + "\n")
		for nid in range(nnode):
			f.write("n " + str(nid) + " 1.5\n" )
		for nid in range(nnode):
			r, c = generatePos(nid, width)
			if(r != 0):
				f.write("e " + str(nid) + " " + str(nid - width) + "\n")
				edgeNum += 1
			if(c != 0):
				f.write("e " + str(nid) + " " + str(nid - 1) + "\n")
				edgeNum += 1
			if(c != width - 1):
				f.write("e " + str(nid) + " " + str(nid + 1) + "\n")
				edgeNum += 1
			if(r != height - 1):
				f.write("e " + str(nid) + " " + str(nid + width) + "\n")
				edgeNum += 1
			if(r % hgap == 0 and c % wgap == 0 and r != 0 and r != height and c != 0 and c != width):
				for hub in hubList:
					if hub != nid:
						f.write("e " + str(nid) + " " + str(hub) + "\n")
						edgeNum += 1
	assert(edgeNum == computeEdgeNum)

def write_rat(filepath, width, height):
	nnode = width * height
	nrat = 4 * nnode
	with open(filepath, 'w') as f:
		f.write(str(nnode) + " " + str(nrat) + "\n")
		f.write("1.0 0.5\n")
		for rid in range(nrat):
			nid = int(random.uniform(0, nnode))
			# nid = nid * width + nid
			# nid = int(random.uniform(0, len(hubList)))
			# nid = hubList[nid]
			tag = "0" if random.uniform(0, 1) >= 0.01 else "1"
			f.write(str(nid) + " " + tag + "\n")

def generatePos(idx, width):
	c = idx % width
	r = idx / width
	return r, c

def generateHub(width, height, wgap, hgap):
	hubList = []
	nnode = width * height
	for nid in range(nnode):
		r, c = generatePos(nid, width)
		if(r % hgap == 0 and c % wgap == 0 and r != 0 and r != height and c != 0 and c != width):
			hubList.append(nid)
	return hubList

if __name__ == '__main__':
	width = 1000
	height = 1000
	hubNum = 1000
	print(width, height, hubNum)
	hubNumSqrt = int(math.sqrt(hubNum))
	wgap = int(width / (1 + hubNumSqrt))
	hgap = int(height / (1 + hubNumSqrt))
	print(hgap)
	graph_file = str(width) + "*" + str(height) + ".graph"
	rat_file = str(width) + "*" + str(height) + ".rats"
	write_graph(graph_file, width, height, wgap, hgap)
	write_rat(rat_file, width, height)