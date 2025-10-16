import os
import numpy as np
import time
from Simulation import *
import copy
import math
from gym import Env
from gym.spaces import Discrete, Box, Dict
from gym.utils import seeding

class Flowsheet(Env):
    def __init__(self, sim, pure, max_iter, inlet_specs):

        # Establish connection with ASPEN
        self.sim = sim

        # Characteristics of the environment
        self.d_actions = 11
        self.pure = pure
        self.max_iter = max_iter
        self.iter = 0
        self.actions_list = []
        self.bzn_pure = False
        self.metan_pure = False
        self.bzn_extra_added = False
        self.value_step = "pre"


        # Declare the initial flowrate conditions
        self.inlet_specs = inlet_specs
        self.Cao = self.inlet_specs[2]["TOL"]
        self.Cbo = self.inlet_specs[2]["HYDROGEN"]

        # Flowsheet
        self.info = {}
        self.infom = {}
        self.avail_actions = np.array(
            [1, # Mixer
             0, # Heater
             0, # Column
             0, # Cooler
             0, # PFR
             0, # PFR adiabatic
             0, # Flash
             0, # Flash with recycle
             0, # Column for CH4
             0, # Column with recycle
             0, # TriColumn 
             ], dtype=np.int32)
        
        
        self.mixer_count = 0
        self.hex_count = 0
        self.cooler_count = 0
        self.pump_count = 0
        self.reac_count = 0
        self.column_count = 0
        self.flash_count = 0

        # Action declaration
        self.action_space = Dict({
            "discrete": Discrete(self.d_actions), 
            "continuous": Box(low=np.zeros(21,), high=np.ones(21,), dtype=np.float32)})


        # Observation
        self.low = np.zeros((7,))
        self.high = np.ones((7,))
        self.observation_space = Box(low=self.low, high=self.high, dtype=np.float32)

        self.bzn_out = 0
        self.metan_out = 0

        self.reset()
        self.seed()
    
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_outputs(self, sout):
        T = sout.get_temp()
        P = sout.get_press()
        Fh = sout.get_molar_flow("HYDROGEN")
        Ft = sout.get_molar_flow("TOL")
        Fbzn = sout.get_molar_flow("BZN")
        Fm = sout.get_molar_flow("METHANE")
        out_list = [T, P, Ft, Fh, Fm, Fbzn]

        return out_list

        

    def step(self, action, sin):
        self.iter += 1
        cost = 0
        
        d_action = action["discrete"]
        c_action = action["continuous"]
        c_action = self.interpolation(np.array(c_action))
        P_hex, T_hex, T_cooler, D1, L1, D2, L2,\
            nstages_cp, dist_rate_cp,\
            nstages_c, dist_rate_c,\
            nstages_cr, dist_rate_cr, rr_cr,\
            nstages_tc, dist_rate_tc,\
            T_flash, P_flash, T_flashr, P_flashr, rr_flash = c_action
      
        
        # ----------------------------------------- Mixer -----------------------------------------
        if d_action == 0:
            self.mixer_count += 1
            self.avail_actions[0] = 0
            self.actions_list.append(f"M{self.mixer_count}")

            mixer = Mixer(f"M{self.mixer_count}", sin)
            sout = mixer.mix()

            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"M{self.mixer_count}"] = self.get_outputs(sout)

                # Costs --> normalized cost approximation 
                f_cost = -0.1
                v_cost = -0 # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
              
        # ----------------------------------------- HEX -----------------------------------------
        elif d_action == 1:
            self.hex_count += 1
            self.actions_list.append(f"HX{self.hex_count}")

            hex = Heater(f"HX{self.hex_count}", T_hex, P_hex, sin)
            sout = hex.heat()
            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"HX{self.hex_count}"] = [T_hex, self.get_outputs(sout)]
                
                # Costs --> normalized cost approximation 
                f_cost = -0.2
                v_cost = -hex.enery_consumption()/(30e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
                

        # ----------------------------------------- Column  -----------------------------------------
        elif d_action == 2:
            
            self.column_count += 1
            self.actions_list.append(f"DC{self.column_count}")

            col = Column(f"DC{self.column_count}", nstages_c, dist_rate_c, 2.5, 1.0, sin)
            
            sout, b = col.distill()
            
            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"DC{self.column_count}"] = [
                    nstages_c, dist_rate_c, self.get_outputs(sout), 
                    self.get_outputs(b)]

                if self.get_outputs(sout)[-1] > 10:
                    self.bzn_out = sout

                # Costs --> normalized cost approximation
                Diam, Height = col.sizing()
                f_cost = -0.2*(1 + self.fixed_cost_column(Diam, Height))
                v_cost = -col.enery_consumption()/(30e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
 
        
        # ----------------------------------------- Cooler -----------------------------------------
        elif d_action == 3:
            self.cooler_count += 1
            self.actions_list.append(f"C{self.cooler_count}")

            cool = Cooler(f"C{self.cooler_count}", T_cooler, sin)

            sout = cool.cool()
            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"C{self.cooler_count}"] = [T_cooler, self.get_outputs(sout)]

                # Costs --> normalized cost approximation 
                f_cost = -0.2
                v_cost = -cool.enery_consumption()/(30e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
        

        # ----------------------------------------- PFR -----------------------------------------
        elif d_action == 4:
            self.reac_count += 1
            self.actions_list.append(f"R{self.reac_count}")
            
            
            pfr = PFR_EX(f"R{self.reac_count}", D1, L1, sin)
    
            sout = pfr.react()
            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"R{self.reac_count}"] = [D1, L1, self.get_outputs(sout)]

                # Costs --> normalized cost approximation
                f_cost = -0.2*(1 + self.fixed_cost_reactor(D1, L1))
                v_cost = -pfr.enery_consumption()/(30e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
        
        # ----------------------------------------- Adiabatic PFR -----------------------------------------
        elif d_action == 5:
            self.reac_count += 1
            self.actions_list.append(f"AR{self.reac_count}")

            pfr_a = PFR_A(f"AR{self.reac_count}", D2, L2, sin)
    
            sout = pfr_a.react()
            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"AR{self.reac_count}"] = [D2,L2, self.get_outputs(sout)]

                # Costs --> normalized cost approximation
                f_cost = -0.2*(1 + self.fixed_cost_reactor(D2, L2))
                v_cost = -0 # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
        
        # ----------------------------------------- Flash -----------------------------------------
        elif d_action == 6:
            self.flash_count +=1
            self.actions_list.append(f"F{self.flash_count}")
            
            
            flash = Flash(f"F{self.flash_count}",T_flash, P_flash, sin)
            
            v, sout = flash.flash()
                    
            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"F{self.flash_count}"] = [
                    T_flash, P_flash, self.get_outputs(v),
                    self.get_outputs(sout)]

                
                
                # Costs --> normalized cost approximation
                Vin = sin.get_volume_flow()
                V = Vin*0.05/0.2
                f_cost = -0.2*(1 + self.fixed_cost_flash(V))
                v_cost = -flash.enery_consumption()/(30e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost


         # ----------------------------------------- Flash with recycle -----------------------------------------
        elif d_action == 7:
            self.flash_count +=1
            self.actions_list.append(f"FR{self.flash_count}")            

           
            flash = Flash(f"FR{self.flash_count}",T_flashr, P_flashr, sin)
            
            v, sout = flash.flash()
            splitter = Splitter(f"SF{self.flash_count}", rr_flash, v)
            rec2, purge2 = splitter.recycle()

            for i, uo in enumerate(self.actions_list):
                if "M" in uo:
                    self.sim.StreamConnect(self.actions_list[i], rec2.name, "F(IN)")
                    break
                    
            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"FR{self.flash_count}"] = [
                    T_flashr, P_flashr, rr_flash, self.get_outputs(rec2),
                    self.get_outputs(sout)]

                
                
                # Costs --> normalized cost approximatio)
                Vin = sin.get_volume_flow()
                V = Vin*0.05/0.2
                f_cost = -0.2*(1 + self.fixed_cost_flash(V))
                v_cost = -flash.enery_consumption()*rr_flash/(30e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost

        # ----------------------------------------- Column for purge -----------------------------------------
        elif d_action == 8:
                
            self.column_count += 1
            self.actions_list.append(f"PDC{self.column_count}")
            press = sin.get_press()

            if any("M" in action for action in self.actions_list):
                distillation_rate = sin.get_molar_flow("METHANE") + dist_rate_cp
            else:
                distillation_rate = sin.get_molar_flow("METHANE")
                
            col = Column(f"PDC{self.column_count}", nstages_cp, distillation_rate, 1.5, press, sin)
            
            d, sout = col.distill()
            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"PDC{self.column_count}"] = [
                    nstages_cp, distillation_rate, self.get_outputs(d), 
                    self.get_outputs(sout)]

                if self.get_outputs(d)[-2] > 5:
                    self.metan_out = d

                # Costs --> normalized cost approximation
                Diam, Height = col.sizing()
                f_cost = -0.2*(1 + self.fixed_cost_column(Diam, Height))
                v_cost = -col.enery_consumption()/(30e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
        
        # ----------------------------------------- Column with recycle -----------------------------------------
        elif d_action == 9:
            self.column_count += 1
            self.actions_list.append(f"DCR{self.column_count}")

            col = Column(f"DCR{self.column_count}", nstages_cr, dist_rate_cr, 2.5, 1.0, sin)
            
            sout, b = col.distill()
            splitter = Splitter(f"S{self.column_count}", rr_cr, b)
            rec, purge = splitter.recycle()

            for i, uo in enumerate(self.actions_list):
                if "M" in uo:
                    self.sim.StreamConnect(self.actions_list[i], rec.name, "F(IN)")
                    break

            self.sim.EngineRun()


            if self.sim.Convergence():
                self.info[f"DCR{self.column_count}"] = [
                    nstages_cr, dist_rate_cr, rr_cr, self.get_outputs(sout), 
                    self.get_outputs(rec)]

                if self.get_outputs(sout)[-1] > 10:
                    self.bzn_out = sout
        
                self.actions_list.clear()

                # Costs --> normalized cost approximation
                Diam, Height = col.sizing()
                f_cost = -0.2*(1 + self.fixed_cost_column(Diam, Height))
                v_cost = -col.enery_consumption()*rr_cr/(30e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost

        # ----------------------------------------- TriColumn -----------------------------------------
        elif d_action == 10:
                
            self.column_count += 1
            self.actions_list.append(f"TC{self.column_count}")

            col = PartialColumn(f"TC{self.column_count}", nstages_tc, dist_rate_tc, 2.5, 1.0, sin)
            
            sout, b, _ = col.distill()
            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"TC{self.column_count}"] = [
                    nstages_tc, dist_rate_tc, self.get_outputs(sout), 
                    self.get_outputs(b)]

                if self.get_outputs(sout)[-1] > 10:
                    self.bzn_out = sout

                # Costs --> normalized cost approximation
                Diam, Height = col.sizing()
                f_cost = -0.2*(1 + self.fixed_cost_column(Diam, Height))
                v_cost = -col.enery_consumption()/(30e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
                
        # ---------------------------------- Constraints and rewards ----------------------------------     
        if self.sim.Convergence():

            # Constraints

            # Cons 1: (Temperature inside of reactor no greater than 704°C)
            if any("M" in action for action in self.actions_list):
                if d_action in (4, 5) and sout.get_temp() <= 750:
                    bonus_T = 0.2
                else:
                    bonus_T = 0.
            else: 
                if d_action in (4, 5) and sout.get_temp() <= 700:
                    bonus_T = 0.2
                else:
                    bonus_T = 0
            

            # Cons 2: (The proportion of hydrogen to toluene in the reactor should be at least 3:1)    
            if d_action == 7 and (rec2.get_molar_flow("HYDROGEN") + self.Cbo) > 3*self.Cao:
                bonus_F = 0.5
            elif d_action == 6:
                bonus_F = -15.0
            else:
                bonus_F = 0.


            
            # Driving force (reduction of the amount of TOL)
            t_frac_prev = sin.get_molar_flow("TOL")/sin.get_total_molar_flow()
            t_frac = sout.get_molar_flow("TOL")/sout.get_total_molar_flow()
            if not d_action in (6, 7, 8):
                bonus = t_frac_prev - t_frac
            else:
                bonus = 0.

            # Driving force 2 (reduction of the amount of H2)
            b_frac_prev = sin.get_molar_flow("HYDROGEN")/sin.get_total_molar_flow()
            b_frac = sout.get_molar_flow("HYDROGEN")/sout.get_total_molar_flow()
            if d_action in (6, 7):
                bonus2 = 0.8*(b_frac_prev - b_frac)
            else:
                bonus2 = 0.

            # Driving force 3 (reduction of the amount of CH4)
            if d_action == 8:
                bonus3 = 0.4*sout.get_molar_flow("BZN")/ self.Cao
            else:
                bonus3 = 0.
            
             #bonus3 = 0.4*d.get_molar_flow("METHANE") / d.get_total_molar_flow()

                
#            b_frac_prev = sin.get_molar_flow("METHANE")/sin.get_total_molar_flow()
 #           b_frac = sout.get_molar_flow("METHANE")/sout.get_total_molar_flow()
  #          if d_action == 8:
   #             bonus3 = 4.0*(b_frac_prev - b_frac)


            
            # Cons 3. Output purities   
            if not self.metan_pure and self.metan_out != 0:
                w_frac = self.metan_out.get_molar_flow("METHANE")/self.metan_out.get_total_molar_flow()
                self.metan_pure = w_frac >= 0.80
                
            if self.bzn_out != 0:
                self.bzn_pure = self.bzn_out.get_molar_flow("BZN")/self.bzn_out.get_total_molar_flow() >= self.pure

            penalty = 0
            reward_flow = 0
            bzn_extra = 0
            
          
            if self.iter >= self.max_iter:
                self.done = True
                
                if not self.bzn_pure or not self.metan_pure:
                    bzn_frac = sout.get_molar_flow("BZN")/sout.get_total_molar_flow()
                    penalty -= 15*(self.pure - bzn_frac)
            else:
                if self.bzn_pure and self.metan_pure:
                    self.done = True
                    reward_flow += 0.2*(self.max_iter - self.iter)
                    #if any("M" in action for action in self.actions_list):
                        #reward_flow += 0.2*(self.max_iter - self.iter + 1)
                    #else:

            #if self.bzn_out != 0 and self.bzn_out.get_molar_flow("BZN") > 111.8:
             #   bzn_bonus += 0.5
                
            # Reward for more BZN flow
            if self.bzn_pure and not self.bzn_extra_added:
                bzn_extra = 1.2*self.bzn_out.get_molar_flow("BZN") / (self.Cao)
                self.bzn_extra_added = True  # Set the flag to True to indicate that bzn_extra has been added  

       
            reward = cost + bonus + bonus2 + bonus3 + bonus_T + bonus_F + penalty + reward_flow + bzn_extra 
            # reward = bonus_F* (cost + bonus + bonus2 + bonus3 + bonus_T  + penalty + reward_flow + bzn_extra )


            self.state = np.array([
                sout.get_temp()/900,
                sout.get_press()/38,
                sout.get_molar_flow("TOL")/sout.get_total_molar_flow(),
                sout.get_molar_flow("HYDROGEN")/sout.get_total_molar_flow(),
                sout.get_molar_flow("METHANE")/sout.get_total_molar_flow(),
                sout.get_molar_flow("BZN")/sout.get_total_molar_flow(),
                self.iter/self.max_iter])        
        
        
        else:
            self.done = True
            reward = -8

        
        # Return step information
        return self.state, reward, self.done, self.info, sout
        


    def fixed_cost_reactor(self, D, H):
        M_S = 1638.2  # Marshall & Swift equipment index 2018 (1638.2, fixed)
        f_cost = (M_S)/280 * 101.9 * D**1.066 * H**0.802 * (2.18 + 1.15)
        max_cost = (M_S)/280 * 101.9 * 3.5**1.066 * 12**0.802 * (2.18 + 1.15)
        norm_cost = f_cost/max_cost
        return norm_cost
    
    def fixed_cost_column(self, D, H):
        M_S = 1638.2  # Marshall & Swift equipment index 2018 (1638.2, fixed)
        max_H = 1.2*0.61*(25 - 2)
        # Internal costs
        int_cost = (M_S)/280 * D**1.55 * H
        max_int_cost = (M_S)/280 * 2.5**1.55 * max_H
        norm_cost1 = int_cost/max_int_cost

        # External costs
        f_cost = (M_S)/280 * 101.9 * D**1.066 * H**0.802 * (2.18 + 1.15)
        max_cost = (M_S)/280 * 101.9 * 2.5**1.066 * max_H**0.802 * (2.18 + 1.15)
        norm_cost2 = f_cost/max_cost

        return norm_cost1 + norm_cost2

    def fixed_cost_flash(self, V):
        f_cost = (2.25 + 1.82) * (813/397) * 10 ** (3.4974 + 0.4485 * math.log10(V) + 0.1074 * (math.log10(V))**2)
        max_cost = (2.25 + 1.82) * (813/397) * 10 ** (3.4974 + 0.4485 * math.log10(200) + 0.1074 * (math.log10(200))**2)
        norm_cost = f_cost/max_cost
        return norm_cost
    

    def action_masks(self, sin, inlet=None):
        self.masking(sin, inlet)
        v1 = np.ones((self.d_actions,), dtype=np.int32)*self.avail_actions
        mask_vec = np.where(v1 > 0, 1, 0)
        mask_vec = np.array(mask_vec, dtype=bool)
        return mask_vec

    
    def render(self):
        for i in self.info:
            print(f"{i}: {self.info[i]}")


    def interpolation(self, c_action):
        P_hex, T_hex, T_cooler, D1, L1, D2, L2,\
            nstages_cp, dist_rate_cp,\
            nstages_c, dist_rate_c,\
            nstages_cr, dist_rate_cr, rr_cr,\
            nstages_tc, dist_rate_tc,\
            T_flash, P_flash, T_flashr, P_flashr, rr_flash = c_action
             

        P_hex = np.interp(P_hex, [0, 1], (32.0, 38.0))
        T_hex = np.interp(T_hex, [0, 1], (550, 704))
        T_cooler = np.interp(T_cooler, [0, 1], (10, 50))
        D1 = np.interp(D1, [0, 1], (0.5, 3.5))
        L1 = np.interp(L1, [0, 1], (6.5, 12.0))
        D2 = np.interp(D2, [0, 1], (0.5, 3.5))
        L2 = np.interp(L2, [0, 1], (6.5, 12.0))
        nstages_cp = round(np.interp(nstages_cp, [0, 1], [5, 15]) + 0.5)
        dist_rate_cp = np.interp(dist_rate_cp, [0, 1], (0.0, 5.0))
        nstages_c = round(np.interp(nstages_c, [0, 1], [5, 25]) + 0.5)
        dist_rate_c = np.interp(dist_rate_c, [0, 1], (70.0, 130.0))
        nstages_cr = round(np.interp(nstages_cr, [0, 1], [5, 25]) + 0.5)
        dist_rate_cr = np.interp(dist_rate_cr, [0, 1], (70.0, 130.0))
        rr_cr = np.interp(rr_cr, [0, 1], (0.5, 0.95))
        nstages_tc = round(np.interp(nstages_tc, [0, 1], [5, 25]) + 0.5)
        dist_rate_tc = np.interp(dist_rate_tc, [0, 1], (70.0, 130.0))
        T_flash = np.interp(T_flash, [0, 1], (1.0, 50.0))
        P_flash = np.interp(P_flash, [0, 1], (1.0, 38.0))
        T_flashr = np.interp(T_flashr, [0, 1], (1.0, 50.0))
        P_flashr = np.interp(P_flashr, [0, 1], (1.0, 38.0))
        rr_flash = np.interp(rr_flash, [0, 1], (0.5, 0.80))


        y = P_hex, T_hex, T_cooler, D1, L1, D2, L2,\
            nstages_cp, dist_rate_cp,\
            nstages_c, dist_rate_c,\
            nstages_cr, dist_rate_cr, rr_cr,\
            nstages_tc, dist_rate_tc,\
            T_flash, P_flash, T_flashr, P_flashr, rr_flash 
        
        return y


    def reset(self):
        # Reset all instances
        self.iter = 0
        self.sim.Reinitialize()
        T, P, compounds = self.inlet_specs
        Fh, Ft, Fm, Fbzn = compounds["HYDROGEN"], compounds["TOL"], compounds["METHANE"], compounds["BZN"]
        tot_flow = Fh + Ft + Fbzn +Fm
        sin  = Stream("IN", self.inlet_specs)

        self.state = np.array([
                T/900,
                P/38, 
                Fh/tot_flow,
                Ft/tot_flow,
                Fm/tot_flow,
                Fbzn/tot_flow,
                self.iter/self.max_iter])

        self.info.clear()
        self.actions_list.clear()
        self.done = False
        self.avail_actions = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        
        self.value_step = "pre"
        self.mixer_count = 0
        self.hex_count = 0
        self.cooler_count = 0
        self.pump_count = 0
        self.reac_count = 0
        self.column_count = 0
        self.flash_count = 0

        self.bzn_pure = False
        self.bzn_extra_added = False
        self.metan_pure = False

        self.bzn_out = 0
        self.metan_out = 0
        
        return self.state, sin
    

    def masking(self, sin, inlet):

        if inlet:
            T, P, _ = self.inlet_specs
            tol_flow = self.Cao
            conv = 0
            
        else:
            T = sin.get_temp()
            P = sin.get_press()
            tol_flow = sin.get_molar_flow("TOL")
            conv = (self.Cao - tol_flow)/self.Cao
            

        if self.bzn_pure:
            self.value_step = "pure"
        # Preprocess 
        elif T >= 500 and P >= 1 and conv < 0.1:
            self.value_step = "reac"
        elif conv >= 0.75 and self.value_step == "reac":
            self.value_step = "cool" 
        elif self.value_step == "cool":
            self.value_step = "flash"
        elif self.value_step == "flash":
            self.value_step = "predistill"
        elif self.value_step == "predistill" or self.value_step == "distill":
            self.value_step = "distill"
     
        
         
      
        # Preparation step
        if self.value_step == "pre":
            # Heater activation (otherwise error in simulation)
                self.avail_actions[1] = 1
            
        elif self.value_step == "hex":
            self.avail_actions = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)    
        
        elif self.value_step == "reac":
            self.avail_actions = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=np.int32)
        
        elif self.value_step == "cool":
            self.avail_actions = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        
        elif self.value_step == "predistill":
            self.avail_actions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.int32)

        elif self.value_step == "distill":
            self.avail_actions = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)

            if any("M" in action for action in self.actions_list):
                if any("DC" in action for action in self.actions_list):
                    self.avail_actions[2] = 0
                    self.avail_actions[9] = 1
                else: 
                    self.avail_actions[8] = 1
            else:
                self.avail_actions[10] = 1
                
                
           
        elif self.value_step == "flash":
            self.avail_actions = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int32)
            
            if any("M" in action for action in self.actions_list):
                self.avail_actions[6] = 0
                self.avail_actions[7] = 1     
            else:
                self.avail_actions[6] = 1

        
        elif self.value_step == "pure":
            self.avail_actions = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)

        return self.avail_actions