def isNaN(num):
        if float('-inf') < float(num) < float('inf'):
            return False 
        else:
            return True