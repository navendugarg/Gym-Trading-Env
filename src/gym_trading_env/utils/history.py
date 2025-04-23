import numpy as np
from typing import Dict, List, Union, Tuple, Any, Optional, Sequence

class History:
    """
    A flexible data structure for storing and retrieving historical trading data.
    
    This class efficiently stores timestep data (actions, observations, rewards, etc.)
    in a fixed-size array and provides various access patterns for retrieving the data.
    
    The history maintains a set of named columns which can be populated from:
    - Scalar values
    - Lists (which get flattened into multiple columns)
    - Dictionaries (which get flattened into multiple columns)
    
    Examples:
        >>> history = History(max_size=1000)
        >>> # Initialize with some data
        >>> history.set(price=100.0, position=[0.5, -0.3], metrics={'sharpe': 1.2, 'return': 0.05})
        >>> # Add data for the next timestep
        >>> history.add(price=101.0, position=[0.6, -0.2], metrics={'sharpe': 1.3, 'return': 0.06})
        >>> # Access data in different ways
        >>> history['price']  # All price values: [100.0, 101.0]
        >>> history['position_0']  # All values for first position: [0.5, 0.6]
        >>> history['metrics_return', 0]  # Return value at timestep 0: 0.05
        >>> history[0]  # All data for timestep 0 as a dictionary
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize a History object with a maximum size.
        
        Parameters
        ----------
        max_size : int, default=10000
            The maximum number of timesteps to store in history.
        """
        self.height = max_size
        self.columns: List[str] = []
        self.width: int = 0
        self.size: int = 0
        self.history_storage: Optional[np.ndarray] = None
        
    def set(self, **kwargs: Any) -> None:
        """
        Initialize the history storage with the first set of data and define the column structure.
        
        This method must be called before adding data with `add()`. It determines the
        column structure based on the provided data and initializes the storage array.
        
        Parameters
        ----------
        **kwargs : Any
            Named data to store. Values can be scalars, lists, or dictionaries.
            - Scalar values create a single column with the provided name
            - Lists create multiple columns named '{name}_{index}'
            - Dictionaries create multiple columns named '{name}_{key}'
        
        Examples
        --------
        >>> history = History()
        >>> history.set(price=100.0, position=[0.5, -0.3], metrics={'sharpe': 1.2, 'return': 0.05})
        >>> # This creates columns: ['price', 'position_0', 'position_1', 'metrics_sharpe', 'metrics_return']
        """
        # Flattening the inputs to put it in np.array
        self.columns = []
        for name, value in kwargs.items():
            if isinstance(value, list):
                self.columns.extend([f"{name}_{i}" for i in range(len(value))])
            elif isinstance(value, dict):
                self.columns.extend([f"{name}_{key}" for key in value.keys()])
            else:
                self.columns.append(name)
        
        self.width = len(self.columns)
        self.history_storage = np.zeros(shape=(self.height, self.width), dtype='O')
        self.size = 0
        self.add(**kwargs)
        
    def add(self, **kwargs: Any) -> None:
        """
        Add a new timestep of data to the history.
        
        The provided data must match the column structure defined when `set()` was called.
        
        Parameters
        ----------
        **kwargs : Any
            Named data to store. The structure must match what was provided to `set()`.
            
        Raises
        ------
        ValueError
            If the provided data doesn't match the column structure defined by `set()`.
            
        Examples
        --------
        >>> history = History()
        >>> history.set(price=100.0, position=[0.5, -0.3])
        >>> history.add(price=101.0, position=[0.6, -0.2])  # Correct
        >>> # history.add(price=102.0)  # Would raise ValueError - missing 'position'
        """
        values = []
        columns = []
        for name, value in kwargs.items():
            if isinstance(value, list):
                columns.extend([f"{name}_{i}" for i in range(len(value))])
                values.extend(value[:])
            elif isinstance(value, dict):
                columns.extend([f"{name}_{key}" for key in value.keys()])
                values.extend(list(value.values()))
            else:
                columns.append(name)
                values.append(value)

        if columns == self.columns:
            self.history_storage[self.size, :] = values
            self.size = min(self.size + 1, self.height)
        else:
            raise ValueError(f"Make sure that your inputs match the initial ones... Initial ones: {self.columns}. New ones: {columns}")
            
    def __len__(self) -> int:
        """
        Return the number of timesteps currently stored in the history.
        
        Returns
        -------
        int
            The number of timesteps in the history.
        """
        return self.size
        
    def __getitem__(self, arg: Union[Tuple[str, int], int, str, List[str]]) -> Any:
        """
        Access history data in various ways.
        
        Parameters
        ----------
        arg : Union[Tuple[str, int], int, str, List[str]]
            Access pattern:
            - (column_name, timestep): Get specific value at specific timestep
            - timestep: Get all values at specific timestep as a dictionary
            - column_name: Get all values for specific column
            - [column_names]: Get all values for multiple columns
            
        Returns
        -------
        Any
            The requested data:
            - Single value if (column, timestep) is requested
            - Dictionary of all values at a timestep
            - Array of all values for a single column
            - 2D array of all values for multiple columns
            
        Raises
        ------
        ValueError
            If the requested column name doesn't exist.
            
        Examples
        --------
        >>> history = History()
        >>> history.set(price=100.0, position=[0.5, -0.3])
        >>> history.add(price=101.0, position=[0.6, -0.2])
        >>> history['price']  # All price values: [100.0, 101.0]
        >>> history[0]  # All data for timestep 0: {'price': 100.0, 'position_0': 0.5, 'position_1': -0.3}
        >>> history['price', 1]  # Price at timestep 1: 101.0
        >>> history[['price', 'position_0']]  # Multiple columns as 2D array
        """
        if isinstance(arg, tuple):
            column, t = arg
            try:
                column_index = self.columns.index(column)
            except ValueError:
                raise ValueError(f"Feature '{column}' does not exist. Available features: {self.columns}")
            return self.history_storage[:self.size][t, column_index]
            
        if isinstance(arg, int):
            t = arg
            return dict(zip(self.columns, self.history_storage[:self.size][t]))
            
        if isinstance(arg, str):
            column = arg
            try:
                column_index = self.columns.index(column)
            except ValueError:
                raise ValueError(f"Feature '{column}' does not exist. Available features: {self.columns}")
            return self.history_storage[:self.size][:, column_index]
            
        if isinstance(arg, list):
            columns = arg
            column_indexes = []
            for column in columns:
                try:
                    column_indexes.append(self.columns.index(column))
                except ValueError:
                    raise ValueError(f"Feature '{column}' does not exist. Available features: {self.columns}")
            return self.history_storage[:self.size, column_indexes]

    def __setitem__(self, arg: Tuple[str, int], value: Any) -> None:
        """
        Set a specific value in the history.
        
        Parameters
        ----------
        arg : Tuple[str, int]
            Tuple of (column_name, timestep) specifying where to set the value
        value : Any
            The value to set
            
        Raises
        ------
        ValueError
            If the specified column doesn't exist.
            
        Examples
        --------
        >>> history = History()
        >>> history.set(price=100.0)
        >>> history['price', 0] = 105.0  # Update the price at timestep 0
        """
        column, t = arg
        try:
            column_index = self.columns.index(column)
        except ValueError:
            raise ValueError(f"Feature '{column}' does not exist. Available features: {self.columns}")
        self.history_storage[:self.size][t, column_index] = value
        
    def to_dict(self) -> Dict[str, List[Any]]:
        """
        Convert the entire history to a dictionary of lists.
        
        Returns
        -------
        Dict[str, List[Any]]
            Dictionary with column names as keys and lists of values as values.
            
        Examples
        --------
        >>> history = History()
        >>> history.set(price=100.0, volume=1000)
        >>> history.add(price=101.0, volume=1200)
        >>> history.to_dict()
        {'price': [100.0, 101.0], 'volume': [1000, 1200]}
        """
        result = {}
        for i, col in enumerate(self.columns):
            result[col] = self.history_storage[:self.size][:, i].tolist()
        return result