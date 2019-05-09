import sc2, random, time
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import (
    NEXUS,
    PROBE,
    PYLON,
    ASSIMILATOR,
    GATEWAY,
    CYBERNETICSCORE,
    STALKER,
    STARGATE,
    VOIDRAY,
)


class Cyrus(sc2.BotAI):

    aggressive_units = {STALKER: {"fight": 15, "defend": 5}, VOIDRAY: {"fight": 8, "defend": 3}}

    def __init__(self):
        self.count = 0
        self.start = time.time()
        self.ITERATION_RATE = 700  # will self adjust
        self.max_workers = 50

    async def on_step(self, iteration):
        self.iteration = iteration
        if time.time() - self.start <= 60:
            self.count += 1
        else:
            self.ITERATION_RATE = self.count
            self.count = 0
            self.start = time.time()
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilates()
        await self.expand()
        await self.offensive_force_buildings()
        await self.build_offensive_force()
        await self.attack()

    async def build_workers(self):
        if len(self.units(NEXUS)) * 16 > len(self.units(PROBE)) and len(self.units(PROBE)) < self.max_workers:
            for nexus in self.units(NEXUS).ready.noqueue:
                if self.can_afford(PROBE):
                    await self.do(nexus.train(PROBE))

    async def build_pylons(self):
        if self.supply_left < 5:
            nexuses = self.units(NEXUS).ready
            if nexuses.exists and self.can_afford(PYLON):
                await self.build(PYLON, near=nexuses.first)

    async def build_assimilates(self):
        for nexus in self.units(NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.5, vaspene).exists:
                    await self.do(worker.build(ASSIMILATOR, vaspene))

    async def expand(self):
        if self.units(NEXUS).amount < (self.iteration / self.ITERATION_RATE) and self.can_afford(NEXUS):
            await self.expand_now()

    async def offensive_force_buildings(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)
            elif len(self.units(GATEWAY)) < (self.iteration / self.ITERATION_RATE) / 2:
                if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near=pylon)
            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(STARGATE)) < (self.iteration / self.ITERATION_RATE) / 2:
                    if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                        await self.build(STARGATE, near=pylon)

    async def build_offensive_force(self):
        for gw in self.units(GATEWAY).ready.noqueue:
            if self.units(STALKER).amount < self.units(VOIDRAY).amount:
                if self.can_afford(STALKER) and self.supply_left > 0:
                    await self.do(gw.train(STALKER))
        for sg in self.units(STARGATE).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.supply_left > 0:
                await self.do(sg.train(VOIDRAY))

    async def attack(self):
        for aggressive_unit, config in self.aggressive_units.items():
            if (
                self.units(aggressive_unit).amount > config["fight"]
                and self.units(aggressive_unit).amount > config["defend"]
            ):
                for s in self.units(STALKER).idle:
                    await self.do(s.attack(random.choice(self.known_enemy_units)))
            elif self.units(aggressive_unit).amount > config["defend"]:
                if len(self.known_enemy_units) > 0:
                    for s in self.units(aggressive_unit).idle:
                        await self.do(s.attack(random.choice(self.known_enemy_units)))


run_game(
    maps.get("AbyssalReefLE"),
    [Bot(Race.Protoss, Cyrus()), Computer(Race.Protoss, Difficulty.Hard)],
    realtime=False,
)
time.sleep(100)
