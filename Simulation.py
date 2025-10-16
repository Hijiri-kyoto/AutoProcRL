from fileinput import filename
import os
from re import A
from tokenize import String
from typing import Union, Dict, Literal
import win32com.client as win32
import numpy as np
import time


class Simulation():
    AspenSimulation = win32.gencache.EnsureDispatch("Apwn.Document")

    def __init__(self, AspenFileName, WorkingDirectoryPath, VISIBILITY=False):
        os.chdir(WorkingDirectoryPath)
        self.AspenSimulation.InitFromArchive2(os.path.abspath(AspenFileName))
        self.AspenSimulation.Visible = VISIBILITY
        self.AspenSimulation.SuppressDialogs = True


    def CloseAspen(self):
        AspenFileName = self.Give_AspenDocumentName()
        self.AspenSimulation.Close(os.path.abspath(AspenFileName))

    def Give_AspenDocumentName(self):
        return self.AspenSimulation.FullName
    
    
    @property
    def BLK(self):
        return self.AspenSimulation.Tree.Elements("Data").Elements("Blocks")

    @property
    def STRM(self):
        return self.AspenSimulation.Tree.Elements("Data").Elements("Streams")

    def EngineRun(self):
        self.AspenSimulation.Run2()

    def EngineStop(self):
        self.AspenSimulation.Stop()

    def EngineReinit(self):
        self.AspenSimulation.Reinit()

    def Convergence(self):
        converged = self.AspenSimulation.Tree.Elements("Data").Elements("Results Summary").Elements(
                           "Run-Status").Elements("Output").Elements("PER_ERROR").Value
        return converged == 0
    
    def StreamConnect(self, Blockname, Streamname, Portname):
        self.BLK.Elements(Blockname).Elements("Ports").Elements(Portname).Elements.Add(Streamname)

    def StreamDisconnect(self, Blockname, Streamname, Portname):
        self.BLK.Elements(Blockname).Elements("Ports").Elements(Portname).Elements.Remove(Streamname)
    
    def Reinitialize(self):
        self.STRM.RemoveAll()
        self.BLK.RemoveAll()
        self.AspenSimulation.Reinit()



class Stream(Simulation):
    def __init__(self, name, inlet=False):
        self.name = name.upper()       
        self.inlet = inlet

        self.StreamPlace()

        if self.inlet:
            self.inlet_stream()
    

    def StreamPlace(self):
        compositstring = self.name + "!" + "MATERIAL"
        self.STRM.Elements.Add(compositstring)

    def StreamDelete(self): 
        self.STRM.Elements.Remove(self.name)
    
    def inlet_stream(self):
        T = self.inlet[0]
        P = self.inlet[1]
        comp = self.inlet[2]

        self.STRM.Elements(self.name).Elements("Input").Elements("TEMP").Elements("MIXED").Value = T
        self.STRM.Elements(self.name).Elements("Input").Elements("PRES").Elements("MIXED").Value = P

        for chemical in comp:
            self.STRM.Elements(self.name).Elements("Input").Elements("FLOW").Elements("MIXED").Elements(
                chemical).Value = comp[chemical]
    
    def get_temp(self):
        return self.STRM.Elements(self.name).Elements("Output").Elements("TEMP_OUT").Elements("MIXED").Value
    
    def get_press(self):
        return self.STRM.Elements(self.name).Elements("Output").Elements("PRES_OUT").Elements("MIXED").Value
    
    def get_molar_flow(self, compound):
        return self.STRM.Elements(self.name).Elements("Output").Elements("MOLEFLOW").Elements("MIXED").Elements(compound).Value
    
    def get_total_molar_flow(self):
        return self.STRM.Elements(self.name).Elements("Output").Elements("MOLEFLMX").Elements("MIXED").Value
    
    def get_vapor_fraction(self):
        return self.STRM.Elements(self.name).Elements("Output").Elements("STR_MAIN").Elements("VFRAC").Elements("MIXED").Value

    def get_volume_flow(self):
    	return self.STRM.Elements(self.name).Elements("Output").Elements("VOLFLMX").Elements("MIXED").Value
    

class Block(Simulation):
    def __init__(self, name, uo):
        self.name = name.upper()
        self.uo = uo

    def BlockCreate(self):
        compositestring = self.name + "!" + self.uo
        self.BLK.Elements.Add(compositestring)

    def BlockDelete(self):
        self.BLK.Elements.Remove(self.name)



# -------------------------------------------------- UNIT OPERATIONS ------------------------------------------------

class Mixer(Block):
    def __init__(self, name, inlet_stream):
        super().__init__(name, "Mixer")
        self.name = name
        self.inlet_stream = inlet_stream

        self.BlockCreate()
        
    
    def mix(self):
        # Inlet connection
        self.StreamConnect(self.name, self.inlet_stream.name, "F(IN)")
        
        self.BLK.Elements(self.name).Elements("Input").Elements("NPHASE").Value = 2
        self.BLK.Elements(self.name).Elements("Input").Elements("PRES").Value = 0
        
        s = Stream(f"{self.name}OUT")
        self.StreamConnect(self.name, s.name, "P(OUT)")
        return s


class Splitter(Block):
    def __init__(self, name, rr, inlet_stream):
        super().__init__(name, "FSplit")

        self.name = name
        self.rr = rr
        self.inlet_stream = inlet_stream
    
        self.BlockCreate()
    
    def recycle(self):
        self.StreamConnect(self.name, self.inlet_stream.name, "F(IN)")

        rec = Stream(f"{self.name}REC")
        s1 = Stream(f"{self.name}PURGE")

        self.StreamConnect(self.name, rec.name, "P(OUT)")
        self.StreamConnect(self.name, s1.name, "P(OUT)")

        self.BLK.Elements(self.name).Elements("Input").Elements("FRAC").Elements(rec.name).Value = self.rr
        return rec, s1


class Vaporizer(Block):
    def __init__(self, name, inlet_stream):
        super().__init__(name, "Heater")

        self.name = name
        self.inlet_stream = inlet_stream

        self.BlockCreate()
    
    def vaporize(self):
        self.StreamConnect(self.name, self.inlet_stream.name, "F(IN)")

        self.BLK.Elements(self.name).Elements("Input").Elements("SPEC_OPT").Value = "PV"
        self.BLK.Elements(self.name).Elements("Input").Elements("VFRAC").Value = 1
        self.BLK.Elements(self.name).Elements("Input").Elements("PRES").Value = 0
         
        
        s = Stream(f"{self.name}OUT")
        self.StreamConnect(self.name, s.name, "P(OUT)")
        return s
    
    def enery_consumption(self):
        q = abs(self.BLK.Elements(self.name).Elements("Output").Elements("QCALC").Value)
        return q



class Heater(Block):
    def __init__(self, name, Temp, press, inlet_stream):
        super().__init__(name, "Heater")

        self.name = name
        self.Temp = Temp
        self.inlet_stream = inlet_stream
        self.press = press
        self.BlockCreate()
    
    def heat(self):
        self.StreamConnect(self.name, self.inlet_stream.name, "F(IN)")

        self.BLK.Elements(self.name).Elements("Input").Elements("SPEC_OPT").Value = "TP"
        self.BLK.Elements(self.name).Elements("Input").Elements("TEMP").Value = self.Temp 
        self.BLK.Elements(self.name).Elements("Input").Elements("PRES").Value = self.press
         
        
        s = Stream(f"{self.name}OUT")
        self.StreamConnect(self.name, s.name, "P(OUT)")
        return s
    
    def enery_consumption(self):
        q = abs(self.BLK.Elements(self.name).Elements("Output").Elements("QCALC").Value)
        return q


class Condenser(Block):
    def __init__(self, name, inlet_stream):
        super().__init__(name, "Heater")

        self.name = name
        self.inlet_stream = inlet_stream

        self.BlockCreate()
    
    def condense(self):
        self.StreamConnect(self.name, self.inlet_stream.name, "F(IN)")

        self.BLK.Elements(self.name).Elements("Input").Elements("SPEC_OPT").Value = "PV"
        self.BLK.Elements(self.name).Elements("Input").Elements("VFRAC").Value = 0
        self.BLK.Elements(self.name).Elements("Input").Elements("PRES").Value = 0
         
        
        s = Stream(f"{self.name}OUT")
        self.StreamConnect(self.name, s.name, "P(OUT)")
        return s
    
    def enery_consumption(self):
        q = abs(self.BLK.Elements(self.name).Elements("Output").Elements("QCALC").Value)
        return q



class Cooler(Block):
    def __init__(self, name, Temp, inlet_stream):
        super().__init__(name, "Heater")

        self.name = name
        self.Temp = Temp
        self.inlet_stream = inlet_stream
    
        self.BlockCreate()
    
    def cool(self):
        self.StreamConnect(self.name, self.inlet_stream.name, "F(IN)")

        self.BLK.Elements(self.name).Elements("Input").Elements("SPEC_OPT").Value = "TP"
        self.BLK.Elements(self.name).Elements("Input").Elements("TEMP").Value = self.Temp 
        self.BLK.Elements(self.name).Elements("Input").Elements("PRES").Value = 0
         
        
        s = Stream(f"{self.name}OUT")
        self.StreamConnect(self.name, s.name, "P(OUT)")
        return s
    
    def enery_consumption(self):
        q = abs(self.BLK.Elements(self.name).Elements("Output").Elements("QCALC").Value)
        return q



class Pump(Block):
    def __init__(self, name, press, inlet_stream):
        super().__init__(name, "Pump")
        self.name = name
        self.press = press
        self.inlet_stream = inlet_stream

        self.BlockCreate()
    
    def pump(self):
        # Inlet connection
        self.StreamConnect(self.name, self.inlet_stream.name, "F(IN)")
        self.BLK.Elements(self.name).Elements("Input").Elements("OPT_SPEC").Value = "PRES"
        self.BLK.Elements(self.name).Elements("Input").Elements("PRES").Value = self.press

        s = Stream(f"{self.name}OUT")
        self.StreamConnect(self.name, s.name, "P(OUT)")
        return s

    def enery_consumption(self):
        q = abs(self.BLK.Elements(self.name).Elements("Output").Elements("WNET").Value)
        return q



class PFR(Block):
    def __init__(self, name, D, L, T, inlet_stream):
        super().__init__(name, "RPlug")
        self.name = name
        self.inlet_stream = inlet_stream
        self.D = D
        self.L = L
        self.T = T
        
        self.BlockCreate()
    
    def react(self):
        # Inlet connection
        self.StreamConnect(self.name, self.inlet_stream.name, "F(IN)")

        # Reactors specifications
        self.BLK.Elements(self.name).Elements("Input").Elements("TYPE").Value = "T-SPEC"
        self.BLK.Elements(self.name).Elements("Input").Elements("OPT_TSPEC").Value = "CONST-TEMP"
        self.BLK.Elements(self.name).Elements("Input").Elements("REAC_TEMP").Value = self.T

        # Sizing
        self.BLK.Elements(self.name).Elements("Input").Elements("NPHASE").Value = 1
        self.BLK.Elements(self.name).Elements("Input").Elements("LENGTH").Value = self.L
        self.BLK.Elements(self.name).Elements("Input").Elements("DIAM").Value = self.D

        # Reaction
        nodes = self.AspenSimulation.Application.Tree.FindNode(f"/Data/Blocks/{self.name}/Input/RXN_ID").Elements
        nodes.InsertRow(1, nodes.Count)
        nodes(nodes.Count - 1).Value = "R-1"

        # Pressure
        self.BLK.Elements(self.name).Elements("Input").Elements("OPT_PDROP").Value = "SPECIFIED"
        self.BLK.Elements(self.name).Elements("Input").Elements("PDROP").Value = 0
        
        # Catalyst 
        self.BLK.Elements(self.name).Elements("Input").Elements("CAT_PRESENT").Value = "NO"
        
        s = Stream(f"{self.name}OUT")
        self.StreamConnect(self.name, s.name, "P(OUT)")
        return s

    def enery_consumption(self):
        q = abs(self.BLK.Elements(self.name).Elements("Output").Elements("QCALC").Value)
        return q


class PFR_EX(Block):
    def __init__(self, name, D, L, inlet_stream):
        super().__init__(name, "RPlug")
        self.name = name
        self.inlet_stream = inlet_stream
        self.D = D
        self.L = L

        self.BlockCreate()
    
    def react(self):
        # Inlet connection
        self.StreamConnect(self.name, self.inlet_stream.name, "F(IN)")

        # Reactors specifications
        self.BLK.Elements(self.name).Elements("Input").Elements("TYPE").Value = "TCOOL-SPEC"
        self.BLK.Elements(self.name).Elements("Input").Elements("U").Value = 80
        self.BLK.Elements(self.name).Elements("Input").Elements("CTEMP").Value = 550

        # Sizing
        self.BLK.Elements(self.name).Elements("Input").Elements("NPHASE").Value = 1
        self.BLK.Elements(self.name).Elements("Input").Elements("LENGTH").Value = self.L
        self.BLK.Elements(self.name).Elements("Input").Elements("DIAM").Value = self.D

        # Reaction
        nodes = self.AspenSimulation.Application.Tree.FindNode(f"/Data/Blocks/{self.name}/Input/RXN_ID").Elements
        nodes.InsertRow(1, nodes.Count)
        nodes(nodes.Count - 1).Value = "R-1"

        # Pressure
        self.BLK.Elements(self.name).Elements("Input").Elements("OPT_PDROP").Value = "SPECIFIED"
        self.BLK.Elements(self.name).Elements("Input").Elements("PDROP").Value = 0
        
        # Catalyst 
        self.BLK.Elements(self.name).Elements("Input").Elements("CAT_PRESENT").Value = "NO"


        s = Stream(f"{self.name}OUT")
        self.StreamConnect(self.name, s.name, "P(OUT)")
        return s

    def enery_consumption(self):
        q = abs(self.BLK.Elements(self.name).Elements("Output").Elements("QCALC").Value)
        return q




class PFR_A(Block):
    def __init__(self, name, D, L, inlet_stream):
        super().__init__(name, "RPlug")
        self.name = name
        self.inlet_stream = inlet_stream
        self.D = D
        self.L = L

        self.BlockCreate()
    
    def react(self):
        # Inlet connection
        self.StreamConnect(self.name, self.inlet_stream.name, "F(IN)")

        # Reactors specifications
        self.BLK.Elements(self.name).Elements("Input").Elements("TYPE").Value = "ADIABATIC"
       
        # Sizing
        self.BLK.Elements(self.name).Elements("Input").Elements("NPHASE").Value = 1
        self.BLK.Elements(self.name).Elements("Input").Elements("LENGTH").Value = self.L
        self.BLK.Elements(self.name).Elements("Input").Elements("DIAM").Value = self.D

        # Reaction
        nodes = self.AspenSimulation.Application.Tree.FindNode(f"/Data/Blocks/{self.name}/Input/RXN_ID").Elements
        nodes.InsertRow(1, nodes.Count)
        nodes(nodes.Count - 1).Value = "R-1"

        # Pressure
        self.BLK.Elements(self.name).Elements("Input").Elements("OPT_PDROP").Value = "SPECIFIED"
        self.BLK.Elements(self.name).Elements("Input").Elements("PDROP").Value = 0    
        
        # Catalyst 
        self.BLK.Elements(self.name).Elements("Input").Elements("CAT_PRESENT").Value = "NO"
    


        s = Stream(f"{self.name}OUT")
        self.StreamConnect(self.name, s.name, "P(OUT)")
        return s



class Column(Block):
    def __init__(self, name, nstages, dist_rate, reflux_ratio, press, inlet_stream):
        super().__init__(name, "Radfrac")
        self.name = name
        self.nstages = nstages
        self.dist_rate = dist_rate
        self.reflux_ratio = reflux_ratio
        self.press = press
        self.inlet_stream = inlet_stream
    
        self.BlockCreate()
    
    def distill(self):
        self.StreamConnect(self.name, self.inlet_stream.name, "F(IN)")
                
        # Configuration
        self.BLK.Elements(self.name).Elements("Input").Elements("CALC_MODE").Value = "EQUILIBRIUM"
        self.BLK.Elements(self.name).Elements("Input").Elements("NSTAGE").Value = self.nstages
        self.BLK.Elements(self.name).Elements("Input").Elements("CONDENSER").Value = "TOTAL"
        self.BLK.Elements(self.name).Elements("Input").Elements("REBOILER").Value = "KETTLE"
        self.BLK.Elements(self.name).Elements("Input").Elements("NO_PHASE").Value = 2
        self.BLK.Elements(self.name).Elements("Input").Elements("CONV_METH").Value = "STANDARD" 
        self.BLK.Elements(self.name).Elements("Input").Elements("BASIS_D").Value = self.dist_rate
        self.BLK.Elements(self.name).Elements("Input").Elements("BASIS_RR").Value = self.reflux_ratio



        # Streams
        self.BLK.Elements(self.name).Elements("Input").Elements("FEED_STAGE").Elements(self.inlet_stream.name).Value = round(self.nstages/2, 0)
        self.BLK.Elements(self.name).Elements("Input").Elements("FEED_CONVE2").Elements(self.inlet_stream.name).Value = "ABOVE-STAGE"

        # Pressure
        self.BLK.Elements(self.name).Elements("Input").Elements("PRES1").Value = self.press

        # Convergence
        self.BLK.Elements(self.name).Elements("Input").Elements("MAXOL").Value = 200

        # Tray sizing
        self.BLK.Elements(self.name).Elements("Subobjects").Elements("Tray Sizing").Elements.Add("1")
      
        self.BLK.Elements(self.name).Elements("Subobjects").Elements("Tray Sizing").Elements("1").Elements("Input").Elements("TS_STAGE1").Elements("1").Value = 2
        self.BLK.Elements(self.name).Elements("Subobjects").Elements("Tray Sizing").Elements("1").Elements("Input").Elements("TS_STAGE2").Elements("1").Value = self.nstages - 1
        self.BLK.Elements(self.name).Elements("Subobjects").Elements("Tray Sizing").Elements("1").Elements("Input").Elements("TS_TRAYTYPE").Elements("1").Value = "SIEVE"


        d = Stream(f"{self.name}DOUT")
        self.StreamConnect(self.name, d.name, "LD(OUT)")
        b = Stream(f"{self.name}BOUT")
        self.StreamConnect(self.name, b.name, "B(OUT)")
        return d, b
    
    def enery_consumption(self):
        q1 = abs(self.BLK.Elements(self.name).Elements("Output").Elements("COND_DUTY").Value)
        q2 = abs(self.BLK.Elements(self.name).Elements("Output").Elements("REB_DUTY").Value)
        return q1 + q2
    
    def sizing(self):
        D = self.BLK.Elements(self.name).Elements("Subobjects").Elements("Tray Sizing").Elements("1").Elements("Output").Elements("DIAM4").Elements("1").Value
        H = 1.2*0.61*(self.nstages - 2)

        return D, H


class PartialColumn(Block):
    def __init__(self, name, nstages, dist_rate, reflux_ratio, press, inlet_stream):
        super().__init__(name, "Radfrac")
        self.name = name
        self.nstages = nstages
        self.dist_rate = dist_rate
        self.reflux_ratio = reflux_ratio
        self.press = press
        self.inlet_stream = inlet_stream
    
        self.BlockCreate()
    
    def distill(self):
        self.StreamConnect(self.name, self.inlet_stream.name, "F(IN)")
                
        # Configuration
        self.BLK.Elements(self.name).Elements("Input").Elements("CALC_MODE").Value = "EQUILIBRIUM"
        self.BLK.Elements(self.name).Elements("Input").Elements("NSTAGE").Value = self.nstages
        self.BLK.Elements(self.name).Elements("Input").Elements("CONDENSER").Value = "PARTIAL-V-L"
        self.BLK.Elements(self.name).Elements("Input").Elements("REBOILER").Value = "KETTLE"
        self.BLK.Elements(self.name).Elements("Input").Elements("NO_PHASE").Value = 2
        self.BLK.Elements(self.name).Elements("Input").Elements("CONV_METH").Value = "STANDARD" 
        self.BLK.Elements(self.name).Elements("Input").Elements("BASIS_D").Value = self.dist_rate
        self.BLK.Elements(self.name).Elements("Input").Elements("BASIS_RR").Value = self.reflux_ratio

        # Condenser
        self.BLK.Elements(self.name).Elements("Input").Elements("BASIS_RDV").Value = 0.05

        # Streams
        self.BLK.Elements(self.name).Elements("Input").Elements("FEED_STAGE").Elements(self.inlet_stream.name).Value = round(self.nstages/2, 0)
        self.BLK.Elements(self.name).Elements("Input").Elements("FEED_CONVE2").Elements(self.inlet_stream.name).Value = "ABOVE-STAGE"

        # Pressure
        self.BLK.Elements(self.name).Elements("Input").Elements("PRES1").Value = self.press

        # Convergence
        self.BLK.Elements(self.name).Elements("Input").Elements("MAXOL").Value = 200

        # Tray sizing
        self.BLK.Elements(self.name).Elements("Subobjects").Elements("Tray Sizing").Elements.Add("1")
      
        self.BLK.Elements(self.name).Elements("Subobjects").Elements("Tray Sizing").Elements("1").Elements("Input").Elements("TS_STAGE1").Elements("1").Value = 2
        self.BLK.Elements(self.name).Elements("Subobjects").Elements("Tray Sizing").Elements("1").Elements("Input").Elements("TS_STAGE2").Elements("1").Value = self.nstages - 1
        self.BLK.Elements(self.name).Elements("Subobjects").Elements("Tray Sizing").Elements("1").Elements("Input").Elements("TS_TRAYTYPE").Elements("1").Value = "SIEVE"

        v = Stream(f"{self.name}VOUT")
        self.StreamConnect(self.name, v.name, "VD(OUT)")
        d = Stream(f"{self.name}DOUT")
        self.StreamConnect(self.name, d.name, "LD(OUT)")
        b = Stream(f"{self.name}BOUT")
        self.StreamConnect(self.name, b.name, "B(OUT)")
        return d, b, v
    
    def enery_consumption(self):
        q1 = abs(self.BLK.Elements(self.name).Elements("Output").Elements("COND_DUTY").Value)
        q2 = abs(self.BLK.Elements(self.name).Elements("Output").Elements("REB_DUTY").Value)
        return q1 + q2
    
    def sizing(self):
        D = self.BLK.Elements(self.name).Elements("Subobjects").Elements("Tray Sizing").Elements("1").Elements("Output").Elements("DIAM4").Elements("1").Value
        H = 1.2*0.61*(self.nstages - 2)

        return D, H


class TriColumn(Block):
    def __init__(self, name, nstages, dist_rate, reflux_ratio, press, mid_rate, inlet_stream):
        super().__init__(name, "Radfrac")
        self.name = name
        self.nstages = nstages
        self.dist_rate = dist_rate
        self.reflux_ratio = reflux_ratio
        self.press = press
        self.mid_rate = mid_rate
        self.inlet_stream = inlet_stream
    
        self.BlockCreate()
    
    def distill(self):
        self.StreamConnect(self.name, self.inlet_stream.name, "F(IN)")
        
        # Configuration
        self.BLK.Elements(self.name).Elements("Input").Elements("CALC_MODE").Value = "EQUILIBRIUM"
        self.BLK.Elements(self.name).Elements("Input").Elements("NSTAGE").Value = self.nstages
        self.BLK.Elements(self.name).Elements("Input").Elements("CONDENSER").Value = "TOTAL"
        self.BLK.Elements(self.name).Elements("Input").Elements("REBOILER").Value = "KETTLE"
        self.BLK.Elements(self.name).Elements("Input").Elements("NO_PHASE").Value = 2
        self.BLK.Elements(self.name).Elements("Input").Elements("CONV_METH").Value = "STANDARD" 
        self.BLK.Elements(self.name).Elements("Input").Elements("BASIS_D").Value = self.dist_rate
        self.BLK.Elements(self.name).Elements("Input").Elements("BASIS_RR").Value = self.reflux_ratio

        # Streams
        self.BLK.Elements(self.name).Elements("Input").Elements("FEED_STAGE").Elements(self.inlet_stream.name).Value = round(self.nstages/3, 0)
        self.BLK.Elements(self.name).Elements("Input").Elements("FEED_CONVE2").Elements(self.inlet_stream.name).Value = "ABOVE-STAGE"
                
        # Pressure
        self.BLK.Elements(self.name).Elements("Input").Elements("PRES1").Value = self.press

        # Convergence
        self.BLK.Elements(self.name).Elements("Input").Elements("MAXOL").Value = 200

        # Tray sizing
        self.BLK.Elements(self.name).Elements("Subobjects").Elements("Tray Sizing").Elements.Add("1")
      
        self.BLK.Elements(self.name).Elements("Subobjects").Elements("Tray Sizing").Elements("1").Elements("Input").Elements("TS_STAGE1").Elements("1").Value = 2
        self.BLK.Elements(self.name).Elements("Subobjects").Elements("Tray Sizing").Elements("1").Elements("Input").Elements("TS_STAGE2").Elements("1").Value = self.nstages - 1
        self.BLK.Elements(self.name).Elements("Subobjects").Elements("Tray Sizing").Elements("1").Elements("Input").Elements("TS_TRAYTYPE").Elements("1").Value = "SIEVE"


        d = Stream(f"{self.name}DOUT")
        self.StreamConnect(self.name, d.name, "LD(OUT)")
        mid = Stream(f"{self.name}MOUT")
        self.StreamConnect(self.name, mid.name, "SP(OUT)")
        b = Stream(f"{self.name}BOUT")
        self.StreamConnect(self.name, b.name, "B(OUT)")

        self.BLK.Elements(self.name).Elements("Input").Elements("PROD_PHASE").Elements(mid.name).Value = "L"
        self.BLK.Elements(self.name).Elements("Input").Elements("PROD_STAGE").Elements(mid.name).Value = round(self.nstages/2, 0)
        self.BLK.Elements(self.name).Elements("Input").Elements("PROD_FLOW").Elements(mid.name).Value = self.mid_rate


        return d, mid, b

    def enery_consumption(self):
        q1 = abs(self.BLK.Elements(self.name).Elements("Output").Elements("COND_DUTY").Value)
        q2 = abs(self.BLK.Elements(self.name).Elements("Output").Elements("REB_DUTY").Value)
        return q1 + q2
    
    def sizing(self):
        D = self.BLK.Elements(self.name).Elements("Subobjects").Elements("Tray Sizing").Elements("1").Elements("Output").Elements("DIAM4").Elements("1").Value
        H = 1.2*0.61*(self.nstages - 2)

        return D, H



class Flash(Block):
    def __init__(self, name, Temp, Press, inlet_stream):
        super().__init__(name, "Flash2")
        self.name = name
        self.Temp = Temp
        self.Press = Press
        self.inlet_stream = inlet_stream

        self.BlockCreate()

    def flash(self):
        # Inlet connection
        self.StreamConnect(self.name, self.inlet_stream.name, "F(IN)")

        self.BLK.Elements(self.name).Elements("Input").Elements("TEMP").Value = self.Temp
        self.BLK.Elements(self.name).Elements("Input").Elements("PRES").Value = self.Press

        v = Stream(f"{self.name}VOUT")
        self.StreamConnect(self.name, v.name, "V(OUT)")
        l = Stream(f"{self.name}LOUT")
        self.StreamConnect(self.name, l.name, "L(OUT)")
        return v, l

    def enery_consumption(self):
        q = abs(self.BLK.Elements(self.name).Elements("Output").Elements("QCALC").Value)
        return q

 
    
    
