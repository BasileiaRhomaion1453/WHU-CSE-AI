import time
start=time.time()
map_array = [
    [1,1,1,1,1,1,1,1,1,1],
    [1,0,0,1,0,0,0,1,0,1],
    [1,0,0,1,0,0,0,1,0,1],
    [1,0,0,0,0,1,1,0,0,1],
    [1,0,1,1,1,0,0,0,0,1],
    [1,0,0,0,1,0,0,0,0,1],
    [1,0,1,0,0,0,1,0,0,1],
    [1,0,1,1,1,0,1,1,0,1],
    [1,1,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1]
]
res = []
 
class node:                                   #定义位置结点类    
    def __init__(self, x, y):
        self.x =x
        self.y=y
 
 
def dfs(x, y):
    cur = node(x, y)
    res.append(cur)
    if x == 8 and y == 8:                  #走到出口
        print(len(res))
        for i in range(len(res)):
            print("(" + str(res[i].x)+ "," + str(res[i].y) + ")" , end="-")
        print("")
        return True
    if x == 9 or x == 0 or y == 9 or y == 0 or map_array[x][y] == 1:     #超出范围或遇到墙
        return False
    map_array[x][y] = 1
    if dfs(x+1, y):              #向右走
        return
    if dfs(x-1, y):             #向左走
        return
    if dfs(x, y+1):            #向下走
        return
    if dfs(x, y-1):           #向上走
        return
    map_array[x][y] = 0
    res.pop()

dfs(1,1)
end=time.time()
print('Running time: %s Seconds'%(end-start))