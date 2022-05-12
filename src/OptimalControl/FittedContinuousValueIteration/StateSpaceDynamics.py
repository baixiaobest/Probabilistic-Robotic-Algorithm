class StateSpaceDynamics:
    def __init__(self, A, B, C, D):
        A_check = A.shape[0] == A.shape[1]
        AB_check = A.shape[0] == B.shape[0]
        AC_check = A.shape[0] == C.shape[1]
        CD_check = C.shape[0] == D.shape[0]
        BD_check = B.shape[1] == D.shape[1]

        if not A_check:
            raise ValueError("A check failed")
        if not AB_check:
            raise ValueError("AB check failed")
        if not AC_check:
            raise ValueError("AC check failed")
        if not CD_check:
            raise ValueError("CD check failed")
        if not BD_check:
            raise ValueError("BD check failed")

        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def num_controls(self):
        return self.B.shape[1]

    def num_states(self):
        return self.A.shape[0]

    def dxdt(self, x, u):
        return self.A@x + self.B@u

    def y(self, x, u):
        return self.C@x + self.D@u
