class Package:
    def __init__(self, id, length, width, height, weight, priority, delay_cost):
        self.id = id
        self.length = length
        self.width = width
        self.height = height
        self.weight = weight
        self.priority = priority  # boolean
        self.delay_cost = delay_cost
        self.point1 = (-1, -1, -1)
        self.point2 = (-1, -1, -1)
        self.uld_id = None
