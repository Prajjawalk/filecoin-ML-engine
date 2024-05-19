/**
 * This file is autogenerated by Scaffold-ETH.
 * You should not edit it manually or your changes might be overwritten.
 */
import { GenericContractsDeclaration } from "~~/utils/scaffold-eth/contract";

const deployedContracts = {
  31337: {
    DataLayer: {
      address: "0x5FbDB2315678afecb367f032d93F642f64180aa3",
      abi: [
        {
          inputs: [],
          stateMutability: "nonpayable",
          type: "constructor",
        },
        {
          anonymous: false,
          inputs: [
            {
              indexed: false,
              internalType: "address",
              name: "user",
              type: "address",
            },
            {
              indexed: false,
              internalType: "address",
              name: "provider",
              type: "address",
            },
            {
              indexed: false,
              internalType: "uint256",
              name: "category",
              type: "uint256",
            },
          ],
          name: "NewAnalytics",
          type: "event",
        },
        {
          anonymous: false,
          inputs: [
            {
              indexed: true,
              internalType: "address",
              name: "previousOwner",
              type: "address",
            },
            {
              indexed: true,
              internalType: "address",
              name: "newOwner",
              type: "address",
            },
          ],
          name: "OwnershipTransferred",
          type: "event",
        },
        {
          inputs: [
            {
              internalType: "address payable",
              name: "userAddress",
              type: "address",
            },
            {
              internalType: "bytes32",
              name: "schemaName",
              type: "bytes32",
            },
            {
              internalType: "bytes32[]",
              name: "columns",
              type: "bytes32[]",
            },
            {
              internalType: "uint256[]",
              name: "data",
              type: "uint256[]",
            },
          ],
          name: "addAnalytics",
          outputs: [],
          stateMutability: "payable",
          type: "function",
        },
        {
          inputs: [
            {
              internalType: "bytes32",
              name: "schemaName",
              type: "bytes32",
            },
            {
              internalType: "bytes32[]",
              name: "columns",
              type: "bytes32[]",
            },
            {
              internalType: "enum DataLayer.Category",
              name: "category",
              type: "uint8",
            },
          ],
          name: "addSchema",
          outputs: [],
          stateMutability: "nonpayable",
          type: "function",
        },
        {
          inputs: [
            {
              internalType: "address",
              name: "userAddress",
              type: "address",
            },
          ],
          name: "addUser",
          outputs: [],
          stateMutability: "nonpayable",
          type: "function",
        },
        {
          inputs: [
            {
              internalType: "address",
              name: "",
              type: "address",
            },
          ],
          name: "addressToId",
          outputs: [
            {
              internalType: "uint256",
              name: "",
              type: "uint256",
            },
          ],
          stateMutability: "view",
          type: "function",
        },
        {
          inputs: [
            {
              internalType: "address",
              name: "",
              type: "address",
            },
          ],
          name: "consumerCredits",
          outputs: [
            {
              internalType: "uint256",
              name: "",
              type: "uint256",
            },
          ],
          stateMutability: "view",
          type: "function",
        },
        {
          inputs: [
            {
              internalType: "uint256",
              name: "",
              type: "uint256",
            },
          ],
          name: "dappAnalytics",
          outputs: [
            {
              internalType: "bytes32",
              name: "schemaName",
              type: "bytes32",
            },
            {
              internalType: "enum DataLayer.Category",
              name: "schemaCategory",
              type: "uint8",
            },
          ],
          stateMutability: "view",
          type: "function",
        },
        {
          inputs: [],
          name: "getAllSchemas",
          outputs: [
            {
              components: [
                {
                  internalType: "bytes32",
                  name: "schemaName",
                  type: "bytes32",
                },
                {
                  internalType: "bytes32[]",
                  name: "columns",
                  type: "bytes32[]",
                },
                {
                  internalType: "enum DataLayer.Category",
                  name: "schemaCategory",
                  type: "uint8",
                },
                {
                  internalType: "uint256",
                  name: "totalRecords",
                  type: "uint256",
                },
              ],
              internalType: "struct DataLayer.SchemaDetails[]",
              name: "",
              type: "tuple[]",
            },
          ],
          stateMutability: "view",
          type: "function",
        },
        {
          inputs: [
            {
              internalType: "bytes32",
              name: "schemaName",
              type: "bytes32",
            },
          ],
          name: "getAnalyticsDataBySchemaName",
          outputs: [
            {
              internalType: "uint256[][]",
              name: "",
              type: "uint256[][]",
            },
          ],
          stateMutability: "view",
          type: "function",
        },
        {
          inputs: [
            {
              internalType: "bytes32",
              name: "schemaName",
              type: "bytes32",
            },
          ],
          name: "getColumnsOfSchema",
          outputs: [
            {
              internalType: "bytes32[]",
              name: "",
              type: "bytes32[]",
            },
          ],
          stateMutability: "view",
          type: "function",
        },
        {
          inputs: [
            {
              internalType: "bytes32",
              name: "schemaName",
              type: "bytes32",
            },
            {
              internalType: "address",
              name: "userAddress",
              type: "address",
            },
          ],
          name: "getSchemaAddressToId",
          outputs: [
            {
              internalType: "uint256",
              name: "",
              type: "uint256",
            },
          ],
          stateMutability: "view",
          type: "function",
        },
        {
          inputs: [
            {
              internalType: "bytes32",
              name: "schemaName",
              type: "bytes32",
            },
            {
              internalType: "uint256",
              name: "userId",
              type: "uint256",
            },
          ],
          name: "getSchemaIdToAddress",
          outputs: [
            {
              internalType: "address",
              name: "",
              type: "address",
            },
          ],
          stateMutability: "view",
          type: "function",
        },
        {
          inputs: [],
          name: "getUserActivityMatrix",
          outputs: [
            {
              internalType: "uint256[][]",
              name: "",
              type: "uint256[][]",
            },
          ],
          stateMutability: "view",
          type: "function",
        },
        {
          inputs: [
            {
              internalType: "uint256",
              name: "",
              type: "uint256",
            },
          ],
          name: "idToAddress",
          outputs: [
            {
              internalType: "address",
              name: "",
              type: "address",
            },
          ],
          stateMutability: "view",
          type: "function",
        },
        {
          inputs: [],
          name: "latestIndex",
          outputs: [
            {
              internalType: "uint256",
              name: "",
              type: "uint256",
            },
          ],
          stateMutability: "view",
          type: "function",
        },
        {
          inputs: [],
          name: "owner",
          outputs: [
            {
              internalType: "address",
              name: "",
              type: "address",
            },
          ],
          stateMutability: "view",
          type: "function",
        },
        {
          inputs: [],
          name: "renounceOwnership",
          outputs: [],
          stateMutability: "nonpayable",
          type: "function",
        },
        {
          inputs: [
            {
              internalType: "bytes32",
              name: "",
              type: "bytes32",
            },
          ],
          name: "schemaIndex",
          outputs: [
            {
              internalType: "uint256",
              name: "",
              type: "uint256",
            },
          ],
          stateMutability: "view",
          type: "function",
        },
        {
          inputs: [],
          name: "totalCategories",
          outputs: [
            {
              internalType: "uint256",
              name: "",
              type: "uint256",
            },
          ],
          stateMutability: "view",
          type: "function",
        },
        {
          inputs: [
            {
              internalType: "address",
              name: "newOwner",
              type: "address",
            },
          ],
          name: "transferOwnership",
          outputs: [],
          stateMutability: "nonpayable",
          type: "function",
        },
        {
          inputs: [
            {
              internalType: "address payable",
              name: "userAddress",
              type: "address",
            },
            {
              internalType: "bytes32",
              name: "schemaName",
              type: "bytes32",
            },
            {
              internalType: "bytes32[]",
              name: "columns",
              type: "bytes32[]",
            },
            {
              internalType: "uint256[]",
              name: "data",
              type: "uint256[]",
            },
          ],
          name: "updateAnalytics",
          outputs: [],
          stateMutability: "payable",
          type: "function",
        },
        {
          inputs: [
            {
              internalType: "uint256",
              name: "newReward",
              type: "uint256",
            },
          ],
          name: "updateUserReward",
          outputs: [],
          stateMutability: "nonpayable",
          type: "function",
        },
        {
          inputs: [
            {
              internalType: "uint256",
              name: "",
              type: "uint256",
            },
            {
              internalType: "uint256",
              name: "",
              type: "uint256",
            },
          ],
          name: "userActivityMatrix",
          outputs: [
            {
              internalType: "uint256",
              name: "",
              type: "uint256",
            },
          ],
          stateMutability: "view",
          type: "function",
        },
        {
          inputs: [],
          name: "userRewardPerDatapoint",
          outputs: [
            {
              internalType: "uint256",
              name: "",
              type: "uint256",
            },
          ],
          stateMutability: "view",
          type: "function",
        },
        {
          stateMutability: "payable",
          type: "receive",
        },
      ],
      inheritedFunctions: {
        owner: "@openzeppelin/contracts/access/Ownable.sol",
        renounceOwnership: "@openzeppelin/contracts/access/Ownable.sol",
        transferOwnership: "@openzeppelin/contracts/access/Ownable.sol",
      },
    },
    KNN: {
      address: "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512",
      abi: [
        {
          inputs: [
            {
              internalType: "contract DataLayer",
              name: "_dataLayer",
              type: "address",
            },
          ],
          stateMutability: "nonpayable",
          type: "constructor",
        },
        {
          inputs: [
            {
              internalType: "bytes32",
              name: "schemaName",
              type: "bytes32",
            },
            {
              internalType: "uint256[]",
              name: "row",
              type: "uint256[]",
            },
            {
              internalType: "uint64",
              name: "k",
              type: "uint64",
            },
          ],
          name: "getKNN",
          outputs: [
            {
              internalType: "uint256[][]",
              name: "",
              type: "uint256[][]",
            },
          ],
          stateMutability: "view",
          type: "function",
        },
        {
          inputs: [
            {
              internalType: "uint256[][]",
              name: "analyticsData",
              type: "uint256[][]",
            },
            {
              internalType: "uint256[]",
              name: "row",
              type: "uint256[]",
            },
            {
              internalType: "uint64",
              name: "k",
              type: "uint64",
            },
          ],
          name: "getKNNOffChainData",
          outputs: [
            {
              internalType: "uint256[][]",
              name: "",
              type: "uint256[][]",
            },
          ],
          stateMutability: "pure",
          type: "function",
        },
        {
          inputs: [
            {
              internalType: "address",
              name: "userAddress",
              type: "address",
            },
            {
              internalType: "uint64",
              name: "k",
              type: "uint64",
            },
          ],
          name: "getRecommendedSimilarUsers",
          outputs: [
            {
              internalType: "address[][]",
              name: "",
              type: "address[][]",
            },
          ],
          stateMutability: "view",
          type: "function",
        },
        {
          inputs: [
            {
              internalType: "uint256",
              name: "userIndex",
              type: "uint256",
            },
            {
              internalType: "uint256[][]",
              name: "similarityMatrix",
              type: "uint256[][]",
            },
            {
              internalType: "uint64",
              name: "k",
              type: "uint64",
            },
          ],
          name: "recommend",
          outputs: [
            {
              internalType: "address[][]",
              name: "",
              type: "address[][]",
            },
          ],
          stateMutability: "view",
          type: "function",
        },
        {
          inputs: [],
          name: "totalCategories",
          outputs: [
            {
              internalType: "uint256",
              name: "",
              type: "uint256",
            },
          ],
          stateMutability: "view",
          type: "function",
        },
        {
          inputs: [
            {
              internalType: "uint256",
              name: "",
              type: "uint256",
            },
            {
              internalType: "uint256",
              name: "",
              type: "uint256",
            },
          ],
          name: "userActivityMatrix",
          outputs: [
            {
              internalType: "uint256",
              name: "",
              type: "uint256",
            },
          ],
          stateMutability: "view",
          type: "function",
        },
      ],
      inheritedFunctions: {},
    },
  },
} as const;

export default deployedContracts satisfies GenericContractsDeclaration;
