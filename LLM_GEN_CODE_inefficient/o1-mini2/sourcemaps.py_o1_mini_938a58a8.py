import os
import math
import collections
import json
import shutil

from org.transcrypt import utils

from typing import List, Dict, Any, Optional

'''
A cascaded mini mapping is made as follows:
    - First generated a non-cascaded pretty map
    - After that have the minifier generate a shrink map and load that
    - After that cascade the two to obtain the mini map and save that
'''

# Tools to embed source map info in target code

lineNrLength: int = 6
maxNrOfSourceLinesPerModule: int = 1000000

# Tools to encode and decode numbers as base 64 variable length quantities


class Base64VlqConverter:
    encoding: str
    decoding: Dict[str, int]
    prefabSize: int
    prefab: List[str]

    def __init__(self) -> None:
        self.encoding: str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'  # Forward lookup table, from index to character, 64 symbols in total
        self.decoding: Dict[str, int] = dict((char, i) for i, char in enumerate(self.encoding))  # Enable reverse lookup, so from character to index

        self.prefabSize: int = 256  # The 256 most used vlq's are prefabricated
        self.prefab: List[str] = [self.encode([i], True) for i in range(self.prefabSize)]

    def encode(self, numbers: List[int], init: bool = False) -> str:
        segment: str = ''
        for number in numbers:
            if not init and 0 < number < self.prefabSize:
                field: str = self.prefab[number]
            else:
                signed: str = bin(abs(number))[2:] + ('1' if number < 0 else '0')  # Convert to binary, chop off '0b' and append sign bit
                nChunks: int = math.ceil(len(signed) / 5.0)  # Determine minimum nr of needed 5 bit chunks (2 ^ 5 == 32)
                padded: str = (5 * '0' + signed)[-nChunks * 5 :]  # Pad by prepending zeroes to fit integer nr of chunks
                chunks: List[str] = [
                    ('1' if iChunk else '0') + padded[iChunk * 5 : (iChunk + 1) * 5]
                    for iChunk in range(nChunks - 1, -1, -1)
                ]  # Prefix first chunk with 0, continuation chunks with 1 (continuation bit)
                field = ''.join([self.encoding[int(chunk, 2)] for chunk in chunks])  # Convert each chunk, incl. continuation bit to its encoding
            segment += field
        return segment

    def decode(self, segment: str) -> List[int]:
        numbers: List[int] = []
        accu: int = 0
        weight: int = 1

        sign: int = 1

        for char in segment:
            ordinal: int = self.decoding[char]
            isContinuation: bool = ordinal >= 32

            if isContinuation:
                ordinal -= 32  # Reset continuation bit

            if weight == 1:  # It was the tail of a number
                sign = -1 if ordinal % 2 else 1  # Remember sign
                ordinal //= 2  # Remove sign bit, no matter what it was

            accu += weight * ordinal  # Add new ordinal as currently least significant

            if isContinuation:  # If it's a continuation
                if weight == 1:  # If it's the first continuation it will have the sign bit
                    weight = 16  # So next weight is 16
                else:  # Else it won't have the sign bit:
                    weight *= 32  # So next weight * 32
            else:  # Else  ('no continuation' means 'end of number', since chunks are reversed)
                numbers.append(sign * accu)  # Append accumulated number to results
                accu = 0  # Reset accumulator for next number
                weight = 1  # Reset weight, next number will again start with least significant part

        return numbers


base64VlqConverter: Base64VlqConverter = Base64VlqConverter()

# Tools to create and combine sourcemaps

mapVersion: int = 3
iTargetLine: int
iTargetColumn: int
iSourceIndex: int
iSourceLine: int
iSourceColumn: int
iTargetLine, iTargetColumn, iSourceIndex, iSourceLine, iSourceColumn = range(5)  # Line indexes rather than line numbers are stored


class SourceMapper:
    moduleName: str
    targetDir: str
    minify: bool
    dump: bool
    prettyMappings: List[List[int]]
    shrinkMappings: List[List[int]]
    miniMappings: List[List[int]]
    cascadeMapdumpFile: Optional[Any]

    def __init__(
        self,
        moduleName: str,
        targetDir: str,
        minify: bool,
        dump: bool
    ) -> None:
        self.moduleName: str = moduleName
        self.targetDir: str = targetDir
        self.minify: bool = minify
        self.dump: bool = dump
        self.prettyMappings = []
        self.shrinkMappings = []
        self.miniMappings = []
        self.cascadeMapdumpFile = None

    def generateAndSavePrettyMap(self, sourceLineNrs: List[int]) -> None:
        self.prettyMappings = [[targetLineIndex, 0, 0, sourceLineNr - 1, 0] for targetLineIndex, sourceLineNr in enumerate(sourceLineNrs)]
        self.prettyMappings.sort()

        infix: str = '.pretty' if self.minify else ''

        self.save(self.prettyMappings, infix)

        if self.dump:
            self.dumpMap(self.prettyMappings, infix, '.py')
            self.dumpDeltaMap(self.prettyMappings, infix)

    def cascadeAndSaveMiniMap(self) -> None:
        def getCascadedMapping(shrinkMapping: List[int]) -> List[int]:
            # N.B. self.prettyMappings has to be sorted in advance
            prettyMapping: List[int] = self.prettyMappings[min(shrinkMapping[iSourceLine], len(self.prettyMappings) - 1)]

            result: List[int] = (
                shrinkMapping[: iTargetColumn + 1]  # Target location from shrink mapping
                + prettyMapping[iSourceIndex :]  # Source location from self
            )
            if self.dump and self.cascadeMapdumpFile is not None:
                self.cascadeMapdumpFile.write(f'{result} {shrinkMapping} {prettyMapping}\n')
            return result

        if self.dump:
            self.cascadeMapdumpFile = utils.create(f'{self.targetDir}/{self.moduleName}.cascade_map_dump')

        self.miniMappings = [
            getCascadedMapping(shrinkMapping)
            for shrinkMapping in self.shrinkMappings
        ]
        self.miniMappings.sort()

        self.save(self.miniMappings, '')

        if self.dump and self.cascadeMapdumpFile is not None:
            self.cascadeMapdumpFile.close()

    def loadShrinkMap(self) -> None:
        with open(f'{self.targetDir}/{self.moduleName}.shrink.map') as mapFile:
            rawMap: Dict[str, Any] = json.loads(mapFile.read())

        deltaMappings: List[List[List[int]]] = [
            [base64VlqConverter.decode(segment) for segment in group.split(',')]
            for group in rawMap['mappings'].split(';')
        ]

        '''
        Fields in a delta segment as directly decoded from the output of the minifier:
          index (target line index implicit, is group index)
            0: target column index
            1: source file index        (optional)      (always zero)
            2: source line index        (optional)
            3: source column index      (optional)
            4: name index               (optional)

        Fields in a shrinkMapping:
          index
            0: target line index (is group index, a group represents a target line)
            1: target column index
            2: source file index        (always zero)   (i = 1)
            3: source line index                (i = 2)
            4: source column index              (i = 3)
            5: source name index        (left out)
        '''

        self.shrinkMappings = []
        for groupIndex, deltaGroup in enumerate(deltaMappings):
            for segmentIndex, deltaSegment in enumerate(deltaGroup):
                if deltaSegment:  # Shrink map ends with empty group, i.e. 'holding empty segment'
                    if segmentIndex:
                        new_target_column: int = deltaSegment[0] + self.shrinkMappings[-1][1]
                        self.shrinkMappings.append([groupIndex, new_target_column])
                    else:  # Start of group
                        self.shrinkMappings.append([groupIndex, deltaSegment[0]])  # Absolute target column

                    for i in range(1, 4):  # So i in [1, 2, 3]
                        if groupIndex or segmentIndex:
                            appended_value: int = deltaSegment[i] + self.shrinkMappings[-2][i + 1]
                            self.shrinkMappings[-1].append(appended_value)
                        else:  # Start of map
                            try:
                                self.shrinkMappings[-1].append(deltaSegment[i])  # Absolute file index, source line and source column
                            except:
                                self.shrinkMappings[-1].append(0)  # Shrink map starts with 'A' rather than 'AAAA'
        self.shrinkMappings.sort()  # Sort on target line and inside that on target column

        if self.dump:
            self.dumpMap(self.shrinkMappings, '.shrink', '.py')
            self.dumpDeltaMap(deltaMappings, '.shrink')

    def save(self, mappings: List[List[int]], infix: str) -> None:
        deltaMappings: List[List[int]] = []
        oldMapping: List[int] = [-1, 0, 0, 0, 0]
        for mapping in mappings:
            newGroup: bool = mapping[iTargetLine] != oldMapping[iTargetLine]

            if newGroup:
                deltaMappings.append([])  # Append new group

            deltaMappings[-1].append([])  # Append new segment, one for each mapping

            if newGroup:
                deltaMappings[-1][-1].append(mapping[iTargetColumn])  # Only target column reset for every group
            else:
                deltaMappings[-1][-1].append(mapping[iTargetColumn] - oldMapping[iTargetColumn])  # Others are delta's, so cumulative

            for i in [iSourceIndex, iSourceLine, iSourceColumn]:
                deltaMappings[-1][-1].append(mapping[i] - oldMapping[i])

            oldMapping = mapping

        rawMap: Dict[str, Any] = collections.OrderedDict([
            ('version', mapVersion),
            ('file', f'{self.moduleName}.js'),  # Target
            ('sources', [f'{self.moduleName}{infix}.py']),
            # ('sourcesContent', [None]),
            ('mappings', ';'.join([
                ','.join([
                    base64VlqConverter.encode(segment)
                    for segment in group
                ])
                for group in deltaMappings
            ]))
        ])

        with utils.create(f'{self.targetDir}/{self.moduleName}{infix}.map') as mapFile:
            mapFile.write(json.dumps(rawMap, indent='\t'))

        if self.dump:
            self.dumpMap(mappings, infix, '.py')
            self.dumpDeltaMap(deltaMappings, infix)

    def dumpMap(self, mappings: List[List[int]], infix: str, sourceExtension: str) -> None:
        with utils.create(f'{self.targetDir}/{self.moduleName}{infix}.map_dump') as mapdumpFile:
            mapdumpFile.write(f'mapVersion: {mapVersion}\n\n')
            mapdumpFile.write(f'targetPath: {self.moduleName}.js\n\n')
            mapdumpFile.write(f'sourcePath: {self.moduleName}{infix}{sourceExtension}\n\n')
            mapdumpFile.write('mappings:\n')
            for mapping in mappings:
                mapdumpFile.write('\t{}\n'.format(mapping))

    def dumpDeltaMap(self, deltaMappings: List[List[List[int]]], infix: str) -> None:
        with utils.create(f'{self.targetDir}/{self.moduleName}{infix}.delta_map_dump') as deltaMapdumpFile:
            for group in deltaMappings:
                deltaMapdumpFile.write('(New group) ')
                for segment in group:
                    deltaMapdumpFile.write(f'Segment: {segment}\n')

    def generateMultilevelMap(self) -> None:
        utils.log(False, 'Saving multi-level sourcemap in: {}\n')  # !!!     , 'self.mapPath')
        self.loadShrinkMap()
        self.cascadeAndSaveMiniMap()
