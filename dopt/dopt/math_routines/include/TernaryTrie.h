/** @file
* C++2003 cross-platform implementation of ternary trie as more space efficient implementation compare to r-trie (proposed by R.Sedjvik, J.Nentley in 1990. Space ~4N.)
* - It is fast at least as hashing.
* - Examines only needed characters
* - Support ordered symbol table operations plus others
* - More flexible then red-black BST
*/

#pragma once

#include <vector>
#include <list>
#include <queue>
#include <string>

#include "dopt/system/include/MemoryPool.h"
#include "dopt/system/include/PlatformSpecificMacroses.h"

namespace dopt
{
    /** @class TernaryTrie
    *   @brief Ternary trie proposed in 1990 by R.Sedgewick and J.Bentley. Much more space efficient then rTrie. Most operations ~lg(N) character access.
    */
    template <typename TChar, typename TValue>
    class TernaryTrie
    {
    public:
        typedef const TChar* TString;
        typedef TChar TCharType;
        typedef TValue TValueType;

    private:

        /** Each node has three children
        */
        struct Node
        {
            explicit Node(const TChar& theCharactrer, MemoryPool& theElementsPool)
            : charactrer(theCharactrer)
            , value(TValue())
            , flagHasValue(false)
            , left(nullptr)
            , middle(nullptr)
            , right(nullptr)
            , elementsPool(theElementsPool)
            {
            }

            ~Node() {
                if (left)
                {
                    left->~Node();
                    elementsPool.freeItem(left);
                }

                if (middle)
                {
                    middle->~Node();
                    elementsPool.freeItem(middle);
                }

                if (right)
                {
                    right->~Node();
                    elementsPool.freeItem(right);
                }
            }

            /** Setup or update value assosiated with the node
            * @param theValue value which you want to setup
            */
            void setValue(const TValue& theValue)
            {
                flagHasValue = true;
                value = theValue;
            }

            /** Remove value associated with the node
            */
            void removeValue()
            {
                flagHasValue = false;
            }

            /** Is this node a leaf?
            * @return true if it is so
            */
            bool isLeaf() const
            {
                return left == nullptr && middle == nullptr && right == nullptr;
            }

            /** Does node has a value
            * @return true if it is so
            */
            bool hasValue() const
            {
                return flagHasValue;
            }
            
            const TValue* getValue() const
            {
                if (hasValue())
                    return &value;
                else
                    return nullptr;
            }

            TValue* getValue()
            {
                if (hasValue())
                    return &value;
                else
                    return nullptr;
            }


            TChar charactrer; ///< Assossiated character with the node
            TValue value;      ///< Assossiated value with the node

            Node* left;       ///< Left subtree hold subtree which are less then "current"
            Node* middle;     ///< Middle points to subtree which can hold next part of string
            Node* right;      ///< Right subtree hold subtree which are high then "current"
            
            MemoryPool& elementsPool; ///< Pool for elements of trie that hold this item

            bool flagHasValue;        ///< Flag: Has value or not

        };
        
        Node* root;                   ///< Root for ternary trie
        TValue* emptyKeyValue;        ///< Value for empty key
        MemoryPool trieElementsPool;  ///< Pool for elements of trie
        
    public:

        /** Default constructor
        */
        TernaryTrie()
        : root(nullptr)
        , emptyKeyValue(nullptr)
        , trieElementsPool(sizeof(typename TernaryTrie<TChar, TValue>::Node), 128)
        {}

        /** Default destructor
        */
        ~TernaryTrie()
        {
            // Destruct Key-Value pair for empty key
            delete emptyKeyValue;
            
            // Fast path
            if constexpr (std::is_trivially_copyable<TValue>::value && std::is_trivially_copyable<TChar>::value)
            {
                trieElementsPool.freeAll();                
            }
            else
            {
                if (root)
                {
                    root->~Node();
                    trieElementsPool.freeItem(root);
                }
            }
        }

        /** Move constructor
        */
        TernaryTrie(TernaryTrie&& rhs) noexcept
        : trieElementsPool(rhs.trieElementsPool)
        {
            root = rhs.root;
            rhs.root = nullptr;

            emptyKeyValue = rhs.emptyKeyValue;
            rhs.emptyKeyValue = nullptr;
        }

        /** Move assignment
        */
        TernaryTrie& operator = (TernaryTrie&& rhs) noexcept
        {
            if (this == &rhs)
                return *this;

            clear();

            root = rhs.root;
            rhs.root = nullptr;
                        
            emptyKeyValue = rhs.emptyKeyValue;
            rhs.emptyKeyValue = nullptr;

            trieElementsPool = std::move(rhs.trieElementsPool);
                
            return *this;
        }
        
        /** Copy constructor
        */
        TernaryTrie(const TernaryTrie& rhs) 
        : trieElementsPool(sizeof(typename TernaryTrie<TChar, TValue>::Node), 128)
        {
            if (rhs.emptyKeyValue)
            {
                emptyKeyValue = new TValue(*rhs.emptyKeyValue);
            }
            else
            {
                emptyKeyValue = nullptr;
            }
            
            if (!rhs.root)
            {
                root = nullptr;
                return;
            }

            root = new(trieElementsPool.allocItem()) Node(rhs.root->charactrer, trieElementsPool);
            std::queue<Node*> nodes2copyFrom;
            std::queue<Node*> nodes2copyTo;

            // similar to push_back
            nodes2copyFrom.push(rhs.root);
            nodes2copyTo.push(root);

            while (!nodes2copyFrom.empty())
            {
                Node* src = nodes2copyFrom.front();
                Node* dst = nodes2copyTo.front();

                // similar to pop_front
                nodes2copyFrom.pop();
                nodes2copyTo.pop();

                if (src->hasValue())
                    dst->setValue(*src->getValue());

                if (src->left)
                {
                    dst->left = new(trieElementsPool.allocItem()) Node(src->left->charactrer, trieElementsPool);
                    
                    // similar to push_back
                    nodes2copyFrom.push(src->left);
                    nodes2copyTo.push(dst->left);
                }

                if (src->middle)
                {
                    dst->middle = new(trieElementsPool.allocItem()) Node(src->middle->charactrer, trieElementsPool);
                    
                    // similar to push_back
                    nodes2copyFrom.push(src->middle);
                    nodes2copyTo.push(dst->middle);
                }

                if (src->right)
                {
                    dst->right = new(trieElementsPool.allocItem()) Node(src->right->charactrer, trieElementsPool);
                    
                    // similar to push_back
                    nodes2copyFrom.push(src->right);
                    nodes2copyTo.push(dst->right);
                }                
            }
        }

        /** Assign operator
        */
        TernaryTrie& operator = (const TernaryTrie& rhs)
        {
            if (this == &rhs)
                return *this;

            clear();

            if (!rhs.root)
            {
                // Nothing to assign
                return *this;
            }
            
            // Create empty key-value
            if (rhs.emptyKeyValue)
            {
                emptyKeyValue = new TValue(*rhs.emptyKeyValue);
            }
            else
            {
                emptyKeyValue = nullptr;
            }
            
            // Create root
            root = new(trieElementsPool.allocItem()) Node(rhs.root->charactrer, trieElementsPool);
            
            std::queue<Node*> nodes2copyFrom;
            std::queue<Node*> nodes2copyTo;

            // similar to push_back
            nodes2copyFrom.push(rhs.root);
            nodes2copyTo.push(root);

            while (!nodes2copyFrom.empty())
            {
                Node* src = nodes2copyFrom.front();
                Node* dst = nodes2copyTo.front();

                // similar to pop_front
                nodes2copyFrom.pop();
                nodes2copyTo.pop();

                if (src->hasValue())
                    dst->setValue(*src->getValue());

                if (src->left)
                {
                    dst->left = new(trieElementsPool.allocItem()) Node(src->left->charactrer, trieElementsPool);
                    
                    // similar to push_back
                    nodes2copyFrom.push(src->left);
                    nodes2copyTo.push(dst->left);
                }

                if (src->middle)
                {
                    dst->middle = new(trieElementsPool.allocItem()) Node(src->middle->charactrer, trieElementsPool);
                    
                    // similar to push_back
                    nodes2copyFrom.push(src->middle);
                    nodes2copyTo.push(dst->middle);
                }

                if (src->right)
                {
                    dst->right = new(trieElementsPool.allocItem()) Node(src->right->charactrer, trieElementsPool);
                    
                    // similar to push_back
                    nodes2copyFrom.push(src->right);
                    nodes2copyTo.push(dst->right);
                }
            }

            return *this;
        }

        /** Put or update (key,value) pair in the trie
        * @param key key part
        * @param value value part
        */
        void put(const TChar* key, TValue value)
        {
            if (key[0] == TChar())
            {
                if (emptyKeyValue == nullptr)
                    emptyKeyValue = new TValue(value);
                else
                    *emptyKeyValue = value;                
            }
            else
            {
                // Remarks
                // 1. Key is a sequence of characters from the root to the graph node, which hold value. 
                // 2. Every Key is stored implicitly
                Node* place = nullptr;
                root = modifyInternalRequest(root, key, &place, 0);
                assert(place != nullptr);
                place->setValue(value);
            }
        }

        /** Define does datastructure contains key.
        * @param key key to search
        * @return true if key is inside datastructure
        * @remark Search hit ~L+ln(N).
        * @remark Search miss ~ln(N).
        */
        bool has(const TChar* key) const
        {
            return get(key) != nullptr;
        }

        /** Get value by key. Complexity is ~lg(N) character compares
        * @param key key to find
        * @return pointer to value or null pointer
        */
        TValue* get(const TChar* key) const
        {
            if (key[0] == TChar())
            {
                return emptyKeyValue;
            }
            else
            {
                const Node* x = getInternal(root, key, 0);
                
                if (x == nullptr)
                    return nullptr;
                else
                    return const_cast<TValue*>(x->getValue());
            }
        }

        /** Get value by key. Complexity is ~lg(N) character compares
        * @param key key to find
        * @return pointer to value or null pointer
        */
        TValue* get(std::basic_string_view<TChar> key) const
        {
            if (key.size() == 0)
            {
                return emptyKeyValue;
            }
            else
            {
                const Node* x = getInternal(root, key.data(), key.size(), 0);

                if (x == nullptr)
                {
                    return nullptr;
                }
                else
                {
                    return const_cast<TValue*>(x->getValue());
                }
            }
        }

        /** Get reference to underlying value by key. If key does not exist create it and insert default value for it.
        * @param key key to find
        * @return reference
        */
        TValue& operator[](const TChar* key)
        {
            if (key[0] == TChar())
            {
                if (emptyKeyValue == nullptr)
                    emptyKeyValue = new TValue();

                return *emptyKeyValue;
            }
            else
            {
                Node* place = nullptr;
                root = modifyInternalRequest(root, key, &place, 0);
                assert(place != nullptr);

                if (place->hasValue() == false)
                    place->setValue(TValue());
                
                return *(place->getValue());
            }
        }

        /** Remove reference to specific object by "key"
        * @param key the key which should be removed
        */
        void remove(const TChar* key)
        {
            if (key[0] == TChar())
            {
                delete emptyKeyValue;
                emptyKeyValue = nullptr;
            }
            else
            {
                root = removeInternal(root, key, 0);
            }
        }

        /** Is container empty or not
        * @return true - if container is empty and false - if container is not empty
        */
        bool isEmpty() const
        {
            return root == nullptr && emptyKeyValue == nullptr;
        }

        /** Clean operation. Completely cleanup trie.
        */
        void clear()
        {
            delete emptyKeyValue;
            emptyKeyValue = nullptr;

            // Fast path
            if constexpr (std::is_trivially_copyable<TValue>::value && std::is_trivially_copyable<TChar>::value)
            {
                trieElementsPool.freeAll();
            }
            else
            {
                if (root)
                {
                    root->~Node();
                    trieElementsPool.freeItem(root);
                }
            }

            root = nullptr;
        }
        
        /** Get the longest prefix for specific key
        * @return Number of characters from "key" for which there is something to get
        */
        size_t longestPrefixForKey(const TChar* key) const
        {
            const Node* n = nullptr;
            return longestPrefixForKeyInternal(root, key, 0, 0, &n);
        }

        /**
        * Finds and returns all keys in the trie that start with the given prefix.
        *
        * @param prefix A pointer to the first character of the prefix to search for.
        * @return A vector of OutputString objects containing all keys that match the given prefix.
        */
        template<class OutputString = std::basic_string<TChar>>
        std::vector<OutputString> keysWithPrefix(const TChar* prefix)
        {
            std::vector<OutputString> res;

            // const Node* x = getInternal(root, prefix, 0);
            const Node* x = nullptr;

            if (prefix[0] == TChar())
            {
                if (emptyKeyValue)
                    res.push_back(OutputString());
                x = root;
            }
            else
            {
                x = getInternal(root, prefix, 0);
                if (x != nullptr)
                    x = x->middle;
            }

            OutputString formedString = OutputString();
            
            for (size_t i = 0; prefix[i] != 0; ++i)
                formedString.push_back(prefix[i]);

            collectKeys(x, formedString, res);
            
            return res;
        }

        /**
        * Retrieves a list of all keys stored in the trie.
        *
        * This method collects all the keys present in the ternary trie,
        * including the empty key if it has been inserted, and returns
        * them in a vector of OutputString objects.
        *
        * @return A vector of OutputString objects representing all the keys in the trie.
        */
        template<class OutputString = std::basic_string<TChar>>
        std::vector<OutputString> keys()
        {
            std::vector<OutputString> res;

            if (emptyKeyValue)
                res.push_back(OutputString());
                
            OutputString formedString = OutputString();
            collectKeys(root, formedString, res);
            return res;
        }

    protected:

        /**
         * Gets the internal node corresponding to the given key in the ternary trie.
         * The search starts from the given node and traverses according to the key characters.
         *
         * @param x Initial node from where to start the search.
         * @param key Key to be searched in the ternary trie.
         * @param d Initial index in the key from where to start the comparison.
         * @return Pointer to the node corresponding to the key if found, nullptr otherwise.
        */
        static const Node* getInternal(const Node* restrict_ext x, const TChar* restrict_ext key, size_t d)
        {
            for (;;)
            {
                if (x == nullptr)
                    return nullptr;

                TChar currentSymbol = key[d];
                TChar xCharacter = x->charactrer;

                if (currentSymbol < xCharacter)
                    x = x->left;
                else if (currentSymbol > xCharacter)
                    x = x->right;
                else
                {
                    // We are in the end of the string.
                    if (key[d + 1] == TChar())
                    {
                        return x;
                    }
                    else
                    {
                        x = x->middle;
                        d++;
                    }
                }
            }
        }

        /**
         * Retrieves an internal node from the ternary trie corresponding to a given key.
         *
         * @param x The current node to start the search from.
         * @param key The key for which the corresponding node is to be found.
         * @param keyLength The length of the key.
         * @param d The current depth in the key to be checked.
         * @return A pointer to the found node if the key is present, otherwise nullptr.
         */
        static const Node* getInternal(const Node* restrict_ext x, const TChar* restrict_ext key, size_t keyLength, size_t d)
        {
            TChar currentSymbol = key[d];

            for (;;)
            {
                if (x == nullptr)
                    return nullptr;

                TChar xCharacter = x->charactrer;

                if (currentSymbol < xCharacter)
                    x = x->left;
                else if (currentSymbol > xCharacter)
                    x = x->right;
                else
                {
                    // We are in the end of the string: We have processed [0], [1], ... [d-1] symbols (total d symbols)
                    if (d + 1 == keyLength)
                    {
                        return x;
                    }
                    else
                    {
                        x = x->middle;
                        
                        // Update "d" and "current symbol"
                        currentSymbol = key[d + 1];
                        d = d + 1;
                    }
                }
            }
        }

        /**
         * Determines the longest prefix of the given key that matches in the trie up to a certain depth.
         *
         * @param x Current node in the trie being examined.
         * @param key Key to be searched in the trie.
         * @param d Depth (or index) in the key currently being compared.
         * @param length Length of the longest matching prefix found so far.
         * @param longestPrefixNode Pointer to the node containing the longest matching prefix found so far.
         *
         * @return The length of the longest matching prefix.
         */
        static size_t longestPrefixForKeyInternal(const Node* restrict_ext x, const TChar* restrict_ext key, size_t d, size_t length, const Node** restrict_ext longestPrefixNode)
        {
            if (x == nullptr)
                return length;

            {
                TChar currentSymbol = key[d];
                TChar xCharacter = x->charactrer;
                
                if (currentSymbol < xCharacter)
                    return longestPrefixForKeyInternal(x->left, key, d, length, longestPrefixNode);
                if (currentSymbol > xCharacter)
                    return longestPrefixForKeyInternal(x->right, key, d, length, longestPrefixNode);
                else
                {
                    if (x->hasValue())
                    {
                        length = d + 1;
                        *longestPrefixNode = x;
                    }
                    
                    if (key[d + 1] == TChar())
                        return length;

                    return longestPrefixForKeyInternal(x->middle, key, d + 1, length, longestPrefixNode);
                }
            }
        }

        /**
         * Removes a key from the ternary trie.
         *
         * @param x The node from which to start the removal.
         * @param key The key to be removed from the trie.
         * @param d The current position in the key being processed.
         * @return The modified node after removal.
         */
        Node* removeInternal(Node* restrict_ext x, const TChar* restrict_ext key, size_t d)
        {
            if (x == nullptr)
                return nullptr;
            else
            {
                TChar currentSymbol = key[d];
                TChar xCharacter = x->charactrer;

                if (currentSymbol < xCharacter)
                    x->left = removeInternal(x->left, key, d);
                else if (currentSymbol > xCharacter)
                    x->right = removeInternal(x->right, key, d);
                else
                {
                    if (key[d + 1] == TChar())
                    {
                        x->removeValue();
                    }
                    else
                    {
                        x->middle = removeInternal(x->middle, key, d + 1);
                    }
                }

                if (x && x->isLeaf() && x->hasValue() == false)
                {
                    x->~Node();
                    trieElementsPool.freeItem(x);
                    x = nullptr;
                }
            }

            return x;
        }

        /**
         * Collects all keys in the ternary trie in an inorder fashion.
         *
         * @param x Current Trie node to process.
         * @param prefix Accumulated characters forming the current key.
         * @param q Vector to store the collected keys.
         */
        template<class OutputString>
        static void collectKeys(const Node* restrict_ext x, OutputString& restrict_ext prefix, std::vector<OutputString>& restrict_ext q)
        {
            if (x == nullptr)
                return;

            collectKeys(x->left, prefix, q);          // collect all keys from the "left" part
            
            {
                prefix.push_back(x->charactrer);      // add one more character
                if (x->hasValue())                    // this node has associated value
                {
                    q.push_back(prefix);              // append to result
                }
                collectKeys(x->middle, prefix, q);    // collect all keys from the middle part
                prefix.erase(prefix.size() - 1);      // drop previously appended one more character
            }
            
            collectKeys(x->right, prefix, q);         // collect all keys from the "right" part
        }

        /** Request modification of node which is in the 'key' path
        * @param x current root of subtree
        * @param key whole key which we want to process
        * @param nodeToSetup during recusive calls in this parameter will be saved the address of the node
        * @param d current depth/next symbol! Node* x is correspond to key[d].
        * @return new root for updated subtree
        */
        Node* modifyInternalRequest(Node* restrict_ext x, const TChar* restrict_ext key, Node** restrict_ext nodeToSetup, size_t d)
        {           
            // Special Case when current node is Empty: Fast path to setup all nodes
            if (x == nullptr)
            {
                x = new(trieElementsPool.allocItem()) Node(key[d], trieElementsPool);

                // Eliminate recursion
                for (Node* xCur = x; ; xCur = xCur->middle)
                {
                    if (key[d + 1] == TChar())
                    {
                        *nodeToSetup = xCur;
                        break;
                    }
                    else
                    {
                        xCur->middle = new(trieElementsPool.allocItem()) Node(key[d + 1], trieElementsPool);
                        d = d + 1;
                    }
                }
                
                return x;
            }
            
            // General case
            {
                TChar currentSymbol = key[d];
                TChar xCharacter = x->charactrer;

                {
                    if (currentSymbol < xCharacter)
                    {
                        x->left = modifyInternalRequest(x->left, key, nodeToSetup, d);
                    }
                    else if (currentSymbol > xCharacter)
                    {
                        x->right = modifyInternalRequest(x->right, key, nodeToSetup, d);
                    }
                    else
                    {
                        // Eliminate recursion
                        if (key[d + 1] == TChar())
                        {
                            *nodeToSetup = x;
                        }
                        else
                        {
                            x->middle = modifyInternalRequest(x->middle, key, nodeToSetup, d + 1);
                        }
                    }
                }                
            }
            
            return x;
        }
    };
}
